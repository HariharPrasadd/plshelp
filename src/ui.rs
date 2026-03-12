use crate::*;

#[cfg(unix)]
use std::os::fd::AsRawFd;
#[cfg(unix)]
use termios::{tcsetattr, Termios, ECHO, TCSANOW};
#[cfg(windows)]
use windows_sys::Win32::Foundation::INVALID_HANDLE_VALUE;
#[cfg(windows)]
use windows_sys::Win32::System::Console::{
    GetConsoleMode, GetStdHandle, SetConsoleMode, ENABLE_ECHO_INPUT, STD_INPUT_HANDLE,
};

#[cfg(unix)]
pub(crate) struct TerminalEchoGuard {
    fd: i32,
    original: Termios,
}

#[cfg(unix)]
impl TerminalEchoGuard {
    pub(crate) fn new() -> Option<Self> {
        let fd = stdin().as_raw_fd();
        let mut current = Termios::from_fd(fd).ok()?;
        let original = current.clone();
        current.c_lflag &= !ECHO;
        tcsetattr(fd, TCSANOW, &current).ok()?;
        Some(Self { fd, original })
    }
}

#[cfg(unix)]
impl Drop for TerminalEchoGuard {
    fn drop(&mut self) {
        let _ = tcsetattr(self.fd, TCSANOW, &self.original);
    }
}

#[cfg(windows)]
pub(crate) struct TerminalEchoGuard {
    handle: windows_sys::Win32::Foundation::HANDLE,
    original_mode: u32,
}

#[cfg(windows)]
impl TerminalEchoGuard {
    pub(crate) fn new() -> Option<Self> {
        unsafe {
            let handle = GetStdHandle(STD_INPUT_HANDLE);
            if handle.is_null() || handle == INVALID_HANDLE_VALUE {
                return None;
            }

            let mut original_mode = 0u32;
            if GetConsoleMode(handle, &mut original_mode) == 0 {
                return None;
            }

            let new_mode = original_mode & !ENABLE_ECHO_INPUT;
            if SetConsoleMode(handle, new_mode) == 0 {
                return None;
            }

            Some(Self {
                handle,
                original_mode,
            })
        }
    }
}

#[cfg(windows)]
impl Drop for TerminalEchoGuard {
    fn drop(&mut self) {
        unsafe {
            let _ = SetConsoleMode(self.handle, self.original_mode);
        }
    }
}

#[cfg(not(any(unix, windows)))]
pub(crate) struct TerminalEchoGuard;

#[cfg(not(any(unix, windows)))]
impl TerminalEchoGuard {
    pub(crate) fn new() -> Option<Self> {
        None
    }
}

pub(crate) struct ProgressSpinner {
    done: Arc<AtomicBool>,
    stage: Arc<Mutex<String>>,
    handle: Option<thread::JoinHandle<()>>,
}

impl ProgressSpinner {
    pub(crate) fn new(initial_stage: impl Into<String>) -> Self {
        let done = Arc::new(AtomicBool::new(false));
        let stage = Arc::new(Mutex::new(initial_stage.into()));
        let done_for_spinner = Arc::clone(&done);
        let stage_for_spinner = Arc::clone(&stage);
        let handle = thread::spawn(move || {
            let frames = ["|", "/", "-", "\\"];
            let mut idx = 0usize;
            let mut last_len = 0usize;
            while !done_for_spinner.load(Ordering::Relaxed) {
                let current_stage = stage_for_spinner
                    .lock()
                    .map(|s| s.clone())
                    .unwrap_or_else(|_| String::from("Working"));
                let line = format!("{}... {}", current_stage, frames[idx % frames.len()]);
                let padding = " ".repeat(last_len.saturating_sub(line.len()));
                print!("\r{}{}", line, padding);
                let _ = stdout().flush();
                last_len = line.len();
                idx += 1;
                thread::sleep(Duration::from_millis(150));
            }
        });
        Self {
            done,
            stage,
            handle: Some(handle),
        }
    }
    pub(crate) fn set_stage(&self, message: impl Into<String>) {
        if let Ok(mut stage) = self.stage.lock() {
            *stage = message.into();
        }
    }
    pub(crate) fn finish(mut self) {
        self.done.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
        print!("\r{}\r", " ".repeat(80));
        let _ = stdout().flush();
    }
}

impl Drop for ProgressSpinner {
    fn drop(&mut self) {
        self.done.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
        print!("\r{}\r", " ".repeat(80));
        let _ = stdout().flush();
    }
}
