use crossterm::{
    cursor, execute,
    style::{Color, Print, ResetColor, SetForegroundColor, Attribute, SetAttribute},
    terminal::{self, Clear, ClearType},
};
use std::io::{self, Write};
use std::time::Instant;

const SPINNER_FRAMES: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
const BAR: &str = "━";
const HALF_BAR_LEFT: &str = "╸";
const HALF_BAR_RIGHT: &str = "╺";

pub struct ProgressBar {
    spinner_idx: usize,
    pub current: u64,
    pub estimate: u64,
    last_rendered: Instant,
}

impl ProgressBar {
    pub fn new() -> Self {
        Self {
            spinner_idx: 0,
            current: 0,
            estimate: 1,
            last_rendered: Instant::now(),
        }
    }

    pub fn render(&mut self) -> io::Result<()> {
        let now = Instant::now();
        if now.duration_since(self.last_rendered).as_millis() < 50 {
            return Ok(());
        }
        self.clear()?;
        self.last_rendered = now;

        let width = terminal::size().map(|(w, _)| w as usize).unwrap_or(80);
        let bar_width = width.saturating_sub(SPINNER_FRAMES[0].len() + " 10000 runs ".len() + " 100% ".len());
        let prog_len = (bar_width * 2) as u64 * self.current / self.estimate.max(1);
        let full_bars_len = (prog_len / 2) as usize;

        let mut stdout = io::stdout();
        
        execute!(
            stdout,
            SetForegroundColor(Color::Cyan),
            Print(SPINNER_FRAMES[self.spinner_idx]),
            ResetColor,
            Print(format!(" {:>5} runs ", self.current)),
        )?;
        self.spinner_idx = (self.spinner_idx + 1) % SPINNER_FRAMES.len();

        execute!(stdout, SetForegroundColor(Color::Magenta))?;
        for _ in 0..full_bars_len {
            write!(stdout, "{}", BAR)?;
        }
        if prog_len % 2 == 1 {
            write!(stdout, "{}", HALF_BAR_LEFT)?;
        }

        execute!(
            stdout,
            SetForegroundColor(Color::White),
            SetAttribute(Attribute::Dim)
        )?;
        if prog_len % 2 == 0 {
            write!(stdout, "{}", HALF_BAR_RIGHT)?;
        }
        let remaining = bar_width.saturating_sub(full_bars_len + 1);
        for _ in 0..remaining {
            write!(stdout, "{}", BAR)?;
        }

        execute!(stdout, ResetColor)?;
        let percent = (self.current as f64) * 100.0 / (self.estimate as f64).max(1.0);
        write!(stdout, " {:>3.0}% ", percent)?;
        stdout.flush()?;
        Ok(())
    }

    pub fn clear(&self) -> io::Result<()> {
        let mut stdout = io::stdout();
        execute!(stdout, cursor::MoveToColumn(0), Clear(ClearType::CurrentLine))?;
        Ok(())
    }
}
