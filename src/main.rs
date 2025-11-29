mod progress;

use clap::Parser;
use crossterm::{
    execute,
    style::{Attribute, Color, ResetColor, SetAttribute, SetForegroundColor},
    tty::IsTty,
};
use perf_event::events::Hardware;
use perf_event::Builder;
use progress::ProgressBar;
use std::io::{self, Read, Write};
use std::os::unix::process::ExitStatusExt;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

const MAX_SAMPLES: usize = 10000;

// Track if we've already warned about perf counters being unavailable
static PERF_WARNING_SHOWN: AtomicBool = AtomicBool::new(false);

#[derive(Parser, Debug)]
#[command(name = "repoop")]
#[command(about = "Performance Optimizer Observation Platform\n\nCompares the performance of provided commands using Linux perf events.")]
struct Args {
    /// Duration in milliseconds to sample each command
    #[arg(short, long, default_value = "5000")]
    duration: u64,

    /// Color output mode: auto, never, ansi
    #[arg(long, default_value = "auto")]
    color: String,

    /// Compare performance even if command returns non-zero exit code
    #[arg(short = 'f', long)]
    allow_failures: bool,

    /// Commands to benchmark
    #[arg(required = true)]
    commands: Vec<String>,
}

#[derive(Clone, Copy, Default)]
struct Sample {
    wall_time: u64,
    cpu_cycles: u64,
    instructions: u64,
    cache_references: u64,
    cache_misses: u64,
    branch_misses: u64,
    peak_rss: u64,
}

#[derive(Clone, Copy)]
#[allow(dead_code)]
struct Measurement {
    q1: u64,
    median: u64,
    q3: u64,
    min: u64,
    max: u64,
    mean: f64,
    std_dev: f64,
    outlier_count: u64,
    sample_count: u64,
    unit: Unit,
}

#[derive(Clone, Copy, PartialEq)]
enum Unit {
    Nanoseconds,
    Bytes,
    Count,
}

#[derive(Clone)]
struct Measurements {
    wall_time: Measurement,
    peak_rss: Measurement,
    cpu_cycles: Measurement,
    instructions: Measurement,
    cache_references: Measurement,
    cache_misses: Measurement,
    branch_misses: Measurement,
}

struct BenchCommand {
    raw_cmd: String,
    argv: Vec<String>,
    measurements: Option<Measurements>,
    sample_count: usize,
}

#[derive(Clone, Copy, PartialEq)]
enum ColorMode {
    Auto,
    Never,
    Ansi,
}

impl ColorMode {
    fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "auto" => Some(ColorMode::Auto),
            "never" => Some(ColorMode::Never),
            "ansi" => Some(ColorMode::Ansi),
            _ => None,
        }
    }
}

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {}", e);
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let color_mode = ColorMode::from_str(&args.color).ok_or_else(|| {
        format!(
            "Invalid color mode '{}'. Options: auto, never, ansi",
            args.color
        )
    })?;

    let use_color = match color_mode {
        ColorMode::Auto => io::stdout().is_tty(),
        ColorMode::Never => false,
        ColorMode::Ansi => true,
    };

    let max_duration = Duration::from_millis(args.duration);
    let allow_failures = args.allow_failures;

    let mut commands: Vec<BenchCommand> = args
        .commands
        .iter()
        .map(|cmd| {
            let argv: Vec<String> = cmd.split_whitespace().map(String::from).collect();
            BenchCommand {
                raw_cmd: cmd.clone(),
                argv,
                measurements: None,
                sample_count: 0,
            }
        })
        .collect();

    let total_commands = commands.len();
    let mut bar = ProgressBar::new();
    let mut samples_buf = vec![Sample::default(); MAX_SAMPLES];
    let mut first_measurements: Option<Measurements> = None;

    // Check if perf counters are available before starting benchmarks
    if setup_perf_counters().is_err() && !PERF_WARNING_SHOWN.swap(true, Ordering::Relaxed) {
        eprintln!("warning: perf counters unavailable (permission denied)");
        eprintln!("         CPU metrics will show as 0. To enable, run as root or set:");
        eprintln!("         sudo sysctl kernel.perf_event_paranoid=1");
        eprintln!();
    }

    for (command_idx, command) in commands.iter_mut().enumerate() {
        let command_n = command_idx + 1;
        let min_samples = 3;
        let first_start = Instant::now();
        let mut sample_index = 0;

        while (sample_index < min_samples
            || first_start.elapsed() < max_duration)
            && sample_index < samples_buf.len()
        {
            if use_color {
                bar.render()?;
            }

            let sample = run_sample(&command.argv, allow_failures, command_n, &command.raw_cmd)?;
            samples_buf[sample_index] = sample;
            sample_index += 1;

            if use_color {
                let cur_samples = sample_index as u64;
                let elapsed = first_start.elapsed().as_nanos() as u64;
                let ns_per_sample = elapsed / cur_samples;
                let estimate = if ns_per_sample > 0 {
                    max_duration.as_nanos() as u64 / ns_per_sample
                } else {
                    cur_samples
                };
                bar.estimate = (MAX_SAMPLES as u64).min(estimate.max(cur_samples).max(min_samples as u64));
                bar.current += 1;
            }
        }

        if use_color {
            bar.clear()?;
            bar.current = 0;
            bar.estimate = 1;
        }

        let all_samples = &mut samples_buf[0..sample_index];
        command.sample_count = sample_index;

        let measurements = Measurements {
            wall_time: compute_measurement(all_samples, |s| s.wall_time, Unit::Nanoseconds),
            peak_rss: compute_measurement(all_samples, |s| s.peak_rss, Unit::Bytes),
            cpu_cycles: compute_measurement(all_samples, |s| s.cpu_cycles, Unit::Count),
            instructions: compute_measurement(all_samples, |s| s.instructions, Unit::Count),
            cache_references: compute_measurement(all_samples, |s| s.cache_references, Unit::Count),
            cache_misses: compute_measurement(all_samples, |s| s.cache_misses, Unit::Count),
            branch_misses: compute_measurement(all_samples, |s| s.branch_misses, Unit::Count),
        };

        command.measurements = Some(measurements.clone());

        print_results(
            use_color,
            command,
            command_n,
            total_commands,
            first_measurements.as_ref(),
        )?;

        if command_idx == 0 {
            first_measurements = Some(measurements);
        }
    }

    Ok(())
}

fn run_sample(
    argv: &[String],
    allow_failures: bool,
    command_n: usize,
    raw_cmd: &str,
) -> Result<Sample, Box<dyn std::error::Error>> {
    let start = Instant::now();
    
    // Try to set up perf counters (created fresh for each sample, like poop does)
    let perf_result = setup_perf_counters();
    
    if let Ok((mut cpu_cycles, mut instructions, mut cache_refs, mut cache_misses, mut branch_misses)) = perf_result {
        // Spawn the child - counters will auto-enable on exec due to enable_on_exec(true)
        let mut child = Command::new(&argv[0])
            .args(&argv[1..])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()?;

        let stderr_handle = child.stderr.take();
        let mut stderr_content = String::new();
        if let Some(mut stderr) = stderr_handle {
            let _ = stderr.read_to_string(&mut stderr_content);
        }

        let (status, rusage) = wait4(child.id() as i32)?;
        let wall_time = start.elapsed().as_nanos() as u64;

        if !status.success() && !allow_failures {
            print_error(command_n, raw_cmd, status.code(), &stderr_content);
            std::process::exit(1);
        }

        // ru_maxrss is in kilobytes on Linux
        let peak_rss = (rusage.ru_maxrss as u64) * 1024;

        return Ok(Sample {
            wall_time,
            cpu_cycles: cpu_cycles.read().unwrap_or(0),
            instructions: instructions.read().unwrap_or(0),
            cache_references: cache_refs.read().unwrap_or(0),
            cache_misses: cache_misses.read().unwrap_or(0),
            branch_misses: branch_misses.read().unwrap_or(0),
            peak_rss,
        });
    }

    // Perf wasn't available - just run the command for wall time and RSS
    let mut child = Command::new(&argv[0])
        .args(&argv[1..])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()?;

    let stderr_handle = child.stderr.take();
    let mut stderr_content = String::new();
    if let Some(mut stderr) = stderr_handle {
        let _ = stderr.read_to_string(&mut stderr_content);
    }

    let (status, rusage) = wait4(child.id() as i32)?;
    let wall_time = start.elapsed().as_nanos() as u64;
    let peak_rss = (rusage.ru_maxrss as u64) * 1024;

    if !status.success() && !allow_failures {
        print_error(command_n, raw_cmd, status.code(), &stderr_content);
        std::process::exit(1);
    }

    Ok(Sample {
        wall_time,
        cpu_cycles: 0,
        instructions: 0,
        cache_references: 0,
        cache_misses: 0,
        branch_misses: 0,
        peak_rss,
    })
}

/// Wait for a child process and get its resource usage via wait4 syscall
fn wait4(pid: i32) -> Result<(std::process::ExitStatus, libc::rusage), Box<dyn std::error::Error>> {
    let mut status: i32 = 0;
    let mut rusage: libc::rusage = unsafe { std::mem::zeroed() };
    
    let result = unsafe {
        libc::wait4(pid, &mut status, 0, &mut rusage)
    };
    
    if result == -1 {
        return Err(std::io::Error::last_os_error().into());
    }
    
    // Convert raw status to ExitStatus
    let exit_status = std::process::ExitStatus::from_raw(status);
    
    Ok((exit_status, rusage))
}

fn setup_perf_counters() -> Result<(perf_event::Counter, perf_event::Counter, perf_event::Counter, perf_event::Counter, perf_event::Counter), Box<dyn std::error::Error>> {
    // Note: We use standalone counters instead of a Group because
    // Linux doesn't support inherit + group together
    
    let mut builder = Builder::new().kind(Hardware::CPU_CYCLES);
    builder.inherit(true).enable_on_exec(true);
    let cpu_cycles = builder.build()?;
    
    let mut builder = Builder::new().kind(Hardware::INSTRUCTIONS);
    builder.inherit(true).enable_on_exec(true);
    let instructions = builder.build()?;
    
    let mut builder = Builder::new().kind(Hardware::CACHE_REFERENCES);
    builder.inherit(true).enable_on_exec(true);
    let cache_refs = builder.build()?;
    
    let mut builder = Builder::new().kind(Hardware::CACHE_MISSES);
    builder.inherit(true).enable_on_exec(true);
    let cache_misses = builder.build()?;
    
    let mut builder = Builder::new().kind(Hardware::BRANCH_MISSES);
    builder.inherit(true).enable_on_exec(true);
    let branch_misses = builder.build()?;
    
    Ok((cpu_cycles, instructions, cache_refs, cache_misses, branch_misses))
}

fn print_error(command_n: usize, raw_cmd: &str, code: Option<i32>, stderr: &str) {
    eprintln!(
        "\nerror: Benchmark {} command '{}' failed with exit code {:?}:",
        command_n,
        raw_cmd,
        code
    );
    eprintln!("─────────────────── stderr ───────────────────");
    eprintln!("{}", stderr);
    eprintln!("──────────────────────────────────────────────");
}

#[allow(dead_code)]
fn get_peak_rss() -> u64 {
    0
}

fn compute_measurement<F>(samples: &mut [Sample], get_field: F, unit: Unit) -> Measurement
where
    F: Fn(&Sample) -> u64,
{
    samples.sort_by_key(|s| get_field(s));

    let mut total: u64 = 0;
    let mut min = u64::MAX;
    let mut max = 0u64;

    for s in samples.iter() {
        let v = get_field(s);
        total = total.saturating_add(v);
        min = min.min(v);
        max = max.max(v);
    }

    let mean = total as f64 / samples.len() as f64;
    let mut variance = 0.0;
    for s in samples.iter() {
        let v = get_field(s);
        let delta = v as f64 - mean;
        variance += delta * delta;
    }
    let std_dev = if samples.len() > 1 {
        (variance / (samples.len() - 1) as f64).sqrt()
    } else {
        0.0
    };

    let q1 = get_field(&samples[samples.len() / 4]);
    let q3 = if samples.len() < 4 {
        get_field(&samples[samples.len() - 1])
    } else {
        get_field(&samples[samples.len() - samples.len() / 4])
    };
    let median = get_field(&samples[samples.len() / 2]);

    // Tukey's Fences outliers
    let iqr = (q3 - q1) as f64;
    let low_fence = q1 as f64 - 1.5 * iqr;
    let high_fence = q3 as f64 + 1.5 * iqr;
    let mut outlier_count = 0u64;
    for s in samples.iter() {
        let v = get_field(s) as f64;
        if v < low_fence || v > high_fence {
            outlier_count += 1;
        }
    }

    Measurement {
        q1,
        median,
        q3,
        min,
        max,
        mean,
        std_dev,
        outlier_count,
        sample_count: samples.len() as u64,
        unit,
    }
}

fn print_results(
    use_color: bool,
    command: &BenchCommand,
    command_n: usize,
    total_commands: usize,
    first_measurements: Option<&Measurements>,
) -> io::Result<()> {
    let mut stdout = io::stdout();
    let measurements = command.measurements.as_ref().unwrap();

    // Header
    if use_color {
        execute!(stdout, SetAttribute(Attribute::Bold))?;
    }
    write!(stdout, "Benchmark {}", command_n)?;
    if use_color {
        execute!(stdout, SetAttribute(Attribute::Dim))?;
    }
    write!(stdout, " ({} runs)", command.sample_count)?;
    if use_color {
        execute!(stdout, SetAttribute(Attribute::Reset))?;
    }
    write!(stdout, ":")?;
    for arg in &command.argv {
        write!(stdout, " {}", arg)?;
    }
    writeln!(stdout)?;

    // Column headers
    if use_color {
        execute!(stdout, SetAttribute(Attribute::Bold))?;
    }
    write!(stdout, "  measurement")?;
    write!(stdout, "{:width$}", "", width = 23 - "  measurement".len())?;
    if use_color {
        execute!(stdout, SetForegroundColor(Color::Green), SetAttribute(Attribute::Bold))?;
    }
    write!(stdout, "mean")?;
    if use_color {
        execute!(stdout, ResetColor, SetAttribute(Attribute::Bold))?;
    }
    write!(stdout, " ± ")?;
    if use_color {
        execute!(stdout, SetForegroundColor(Color::DarkGreen))?;
    }
    write!(stdout, "σ")?;
    if use_color {
        execute!(stdout, ResetColor)?;
    }

    if use_color {
        execute!(stdout, SetAttribute(Attribute::Bold))?;
    }
    write!(stdout, "{:>12}", "")?;
    if use_color {
        execute!(stdout, SetForegroundColor(Color::Cyan))?;
    }
    write!(stdout, "min")?;
    if use_color {
        execute!(stdout, ResetColor, SetAttribute(Attribute::Bold))?;
    }
    write!(stdout, " … ")?;
    if use_color {
        execute!(stdout, SetForegroundColor(Color::Magenta))?;
    }
    write!(stdout, "max")?;
    if use_color {
        execute!(stdout, ResetColor)?;
    }

    if use_color {
        execute!(stdout, SetAttribute(Attribute::Bold))?;
    }
    write!(stdout, "{:>12}", "")?;
    if use_color {
        execute!(stdout, SetForegroundColor(Color::Yellow))?;
    }
    write!(stdout, "outliers")?;
    if use_color {
        execute!(stdout, ResetColor)?;
    }

    if total_commands >= 2 {
        if use_color {
            execute!(stdout, SetAttribute(Attribute::Bold))?;
        }
        write!(stdout, "{:>9}delta", "")?;
        if use_color {
            execute!(stdout, ResetColor)?;
        }
    }
    writeln!(stdout)?;

    // Measurements
    let fields = [
        ("wall_time", measurements.wall_time, first_measurements.map(|m| m.wall_time)),
        ("peak_rss", measurements.peak_rss, first_measurements.map(|m| m.peak_rss)),
        ("cpu_cycles", measurements.cpu_cycles, first_measurements.map(|m| m.cpu_cycles)),
        ("instructions", measurements.instructions, first_measurements.map(|m| m.instructions)),
        ("cache_references", measurements.cache_references, first_measurements.map(|m| m.cache_references)),
        ("cache_misses", measurements.cache_misses, first_measurements.map(|m| m.cache_misses)),
        ("branch_misses", measurements.branch_misses, first_measurements.map(|m| m.branch_misses)),
    ];

    for (name, measurement, first_m) in fields {
        print_measurement(use_color, &measurement, name, first_m.as_ref(), total_commands)?;
    }

    stdout.flush()?;
    Ok(())
}

fn print_measurement(
    use_color: bool,
    m: &Measurement,
    name: &str,
    first_m: Option<&Measurement>,
    command_count: usize,
) -> io::Result<()> {
    let mut stdout = io::stdout();

    write!(stdout, "  {}", name)?;
    let spaces = 32usize.saturating_sub("  (mean  ):".len() + name.len() + 2);
    write!(stdout, "{:width$}", "", width = spaces)?;

    // Mean
    if use_color {
        execute!(stdout, SetForegroundColor(Color::Green))?;
    }
    let mean_str = format_unit(m.mean, m.unit, use_color);
    write!(stdout, "{}", mean_str)?;
    if use_color {
        execute!(stdout, ResetColor)?;
    }

    write!(stdout, " ± ")?;

    // Std dev
    if use_color {
        execute!(stdout, SetForegroundColor(Color::DarkGreen))?;
    }
    let std_str = format_unit(m.std_dev, m.unit, use_color);
    write!(stdout, "{}", std_str)?;
    if use_color {
        execute!(stdout, ResetColor)?;
    }

    // Padding
    let used = mean_str.chars().filter(|c| !c.is_control()).count()
        + std_str.chars().filter(|c| !c.is_control()).count()
        + 3;
    let pad = 25usize.saturating_sub(used);
    write!(stdout, "{:width$}", "", width = pad)?;

    // Min
    if use_color {
        execute!(stdout, SetForegroundColor(Color::Cyan))?;
    }
    let min_str = format_unit(m.min as f64, m.unit, use_color);
    write!(stdout, "{}", min_str)?;
    if use_color {
        execute!(stdout, ResetColor)?;
    }

    write!(stdout, " … ")?;

    // Max
    if use_color {
        execute!(stdout, SetForegroundColor(Color::Magenta))?;
    }
    let max_str = format_unit(m.max as f64, m.unit, use_color);
    write!(stdout, "{}", max_str)?;
    if use_color {
        execute!(stdout, ResetColor)?;
    }

    // Outliers
    let outlier_percent = m.outlier_count as f64 / m.sample_count as f64 * 100.0;
    let pad = 10usize;
    write!(stdout, "{:width$}", "", width = pad)?;

    if use_color {
        if outlier_percent >= 10.0 {
            execute!(stdout, SetForegroundColor(Color::Yellow))?;
        } else {
            execute!(stdout, SetAttribute(Attribute::Dim))?;
        }
    }
    write!(stdout, "{:>4} ({:>2.0}%)", m.outlier_count, outlier_percent)?;
    if use_color {
        execute!(stdout, ResetColor, SetAttribute(Attribute::Reset))?;
    }

    // Delta
    if command_count > 1 {
        write!(stdout, "{:>6}", "")?;
        if let Some(f) = first_m {
            // Skip delta calculation if the first measurement mean is 0 (avoid NaN)
            if f.mean == 0.0 {
                if use_color {
                    execute!(stdout, SetAttribute(Attribute::Dim))?;
                }
                write!(stdout, "  N/A")?;
                if use_color {
                    execute!(stdout, SetAttribute(Attribute::Reset))?;
                }
            } else {
                let half = {
                    let z = get_stat_score_95(Some(m.sample_count + f.sample_count - 2));
                    let n1 = m.sample_count as f64;
                    let n2 = f.sample_count as f64;
                    let normer = (1.0 / n1 + 1.0 / n2).sqrt();
                    let numer1 = (n1 - 1.0) * (m.std_dev * m.std_dev);
                    let numer2 = (n2 - 1.0) * (f.std_dev * f.std_dev);
                    let df = n1 + n2 - 2.0;
                    let sp = ((numer1 + numer2) / df).sqrt();
                    (z * sp * normer) * 100.0 / f.mean
                };
                let diff_mean_percent = (m.mean - f.mean) * 100.0 / f.mean;
                let is_sig = {
                    if diff_mean_percent >= 1.0 && (diff_mean_percent - half) >= 1.0 {
                        true
                    } else if diff_mean_percent <= -1.0 && (diff_mean_percent + half) <= -1.0 {
                        true
                    } else {
                        false
                    }
                };

                if m.mean > f.mean {
                    if is_sig {
                        write!(stdout, "💩")?;
                        if use_color {
                            execute!(stdout, SetForegroundColor(Color::Red))?;
                        }
                    } else {
                        if use_color {
                            execute!(stdout, SetAttribute(Attribute::Dim))?;
                        }
                        write!(stdout, "  ")?;
                    }
                    write!(stdout, "+")?;
                } else {
                    if is_sig {
                        if use_color {
                            execute!(stdout, SetForegroundColor(Color::Yellow))?;
                        }
                        write!(stdout, "⚡")?;
                        if use_color {
                            execute!(stdout, SetForegroundColor(Color::Green))?;
                        }
                    } else {
                        if use_color {
                            execute!(stdout, SetAttribute(Attribute::Dim))?;
                        }
                        write!(stdout, "  ")?;
                    }
                    write!(stdout, "-")?;
                }
                write!(stdout, "{:>5.1}% ± {:>4.1}%", diff_mean_percent.abs(), half)?;
                if use_color {
                    execute!(stdout, ResetColor, SetAttribute(Attribute::Reset))?;
                }
            }
        } else {
            if use_color {
                execute!(stdout, SetAttribute(Attribute::Dim))?;
            }
            write!(stdout, "0%")?;
            if use_color {
                execute!(stdout, SetAttribute(Attribute::Reset))?;
            }
        }
    }

    writeln!(stdout)?;
    Ok(())
}

fn format_unit(num: f64, unit: Unit, _use_color: bool) -> String {
    let (val, ustr) = if num >= 1_000_000_000_000.0 {
        (
            num / 1_000_000_000_000.0,
            match unit {
                Unit::Count => "T ",
                Unit::Nanoseconds => "ks",
                Unit::Bytes => "TB",
            },
        )
    } else if num >= 1_000_000_000.0 {
        (
            num / 1_000_000_000.0,
            match unit {
                Unit::Count => "G ",
                Unit::Nanoseconds => "s ",
                Unit::Bytes => "GB",
            },
        )
    } else if num >= 1_000_000.0 {
        (
            num / 1_000_000.0,
            match unit {
                Unit::Count => "M ",
                Unit::Nanoseconds => "ms",
                Unit::Bytes => "MB",
            },
        )
    } else if num >= 1000.0 {
        (
            num / 1000.0,
            match unit {
                Unit::Count => "K ",
                Unit::Nanoseconds => "us",
                Unit::Bytes => "KB",
            },
        )
    } else {
        (
            num,
            match unit {
                Unit::Count => "  ",
                Unit::Nanoseconds => "ns",
                Unit::Bytes => "  ",
            },
        )
    };

    format!("{:>4.1}{}", val, ustr)
}

fn get_stat_score_95(df: Option<u64>) -> f64 {
    const T_TABLE_95_1TO30: [f64; 30] = [
        12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.16,
        2.145, 2.131, 2.12, 2.11, 2.101, 2.093, 2.086, 2.08, 2.074, 2.069, 2.064, 2.06, 2.056,
        2.052, 2.045, 2.048, 2.042,
    ];
    const T_TABLE_95_10S: [f64; 12] = [
        2.228, 2.086, 2.042, 2.021, 2.009, 2.0, 1.994, 1.99, 1.987, 1.984, 1.982, 1.98,
    ];

    if let Some(dff) = df {
        let dfv = dff as usize;
        if dfv >= 1 && dfv <= 30 {
            return T_TABLE_95_1TO30[dfv - 1];
        } else if dfv <= 120 {
            let idx = dfv / 10;
            if idx >= 1 && idx <= 12 {
                return T_TABLE_95_10S[idx - 1];
            }
        }
    }
    1.96
}
