use crate::*;

pub(crate) struct TerminalEchoGuard {
    fd: i32,
    original: Termios,
}

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

impl Drop for TerminalEchoGuard {
    fn drop(&mut self) {
        let _ = tcsetattr(self.fd, TCSANOW, &self.original);
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

