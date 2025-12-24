use fastembed::{TextEmbedding, InitOptions, EmbeddingModel};

/* mixedbread model information

   ModelInfo {
        model: MxbaiEmbedLargeV1Q,
        dim: 1024,
        description: "Quantized Large English embedding model from MixedBreed.ai",
        model_code: "mixedbread-ai/mxbai-embed-large-v1",
        model_file: "onnx/model_quantized.onnx",
        additional_files: [],
        output_key: None,
    },
*/

fn main() {
    println!("Hello, world!");

    let mut model = TextEmbedding::try_new(
       InitOptions::new(EmbeddingModel::MxbaiEmbedLargeV1Q).with_show_download_progress(true),
    ).expect("Could not load model");

     let documents = vec![
    "passage: Hello, World!",
    "query: Hello, World!",
    "passage: This is an example passage.",
    // You can leave out the prefix but it's recommended
    "fastembed-rs is licensed under MIT"
    ];

    // Generate embeddings with the default batch size, 256
    let embeddings = model.embed(documents, None).expect("Could not generate embeddings");

    println!("Embeddings length: {}", embeddings.len()); // -> Embeddings length: 4
}
