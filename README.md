# repoop - Rust Performance Optimizer Observation Platform

A Rust rewrite of [poop](https://github.com/andrewrk/poop) - a Linux command-line tool for comparing performance of commands using `perf_event_open`.

## Features

- Compare performance of multiple commands side-by-side
- Measures wall time, CPU cycles, instructions, cache references/misses, branch misses
- Statistical analysis with mean, standard deviation, min/max, and outlier detection
- Colorful terminal UI with progress bar
- Falls back to wall-time-only mode when perf events aren't available

## Usage

```
Usage: repoop [OPTIONS] <COMMANDS>...

Arguments:
  <COMMANDS>...  Commands to benchmark

Options:
  -d, --duration <DURATION>  Duration in milliseconds to sample each command [default: 5000]
      --color <COLOR>        Color output mode: auto, never, ansi [default: auto]
  -f, --allow-failures       Compare performance even if command returns non-zero exit code
  -h, --help                 Print help
```

### Examples

```bash
# Compare two commands
repoop "echo hello" "sleep 0.01"

# Run for 10 seconds each
repoop -d 10000 "command1" "command2"

# Allow commands that may fail
repoop -f "command_that_might_fail" "another_command"
```

## Building

```bash
cargo build --release
```

The binary will be at `target/release/repoop`.

## Requirements

- Linux (uses `perf_event_open` syscall)
- For full perf counter support, either:
  - Run as root, or
  - Set `kernel.perf_event_paranoid` to a lower value:
    ```bash
    sudo sysctl -w kernel.perf_event_paranoid=1
    ```

Without perf permissions, the tool still works but only measures wall time.

## License

MIT
