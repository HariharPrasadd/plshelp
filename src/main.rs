#[spider::tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let output_json = args.iter().any(|arg| arg == "--json");
    if let Err(err) = plshelp::run(args).await {
        if output_json {
            let payload = serde_json::json!({
                "status": "error",
                "error": format!("{err}"),
            });
            eprintln!(
                "{}",
                serde_json::to_string_pretty(&payload)
                    .unwrap_or_else(|_| format!(r#"{{"status":"error","error":"{}"}}"#, err))
            );
        } else {
            eprintln!("Error: {err}");
        }
        std::process::exit(1);
    }
}
