use fastembed::{TextEmbedding, InitOptions, EmbeddingModel};
use std::{any::type_name, string};

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

// helper function to determine type of input
fn type_of<T>(_: T) -> &'static str {
    type_name::<T>()
}

/// Compute cosine similarity between two vectors
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    
    dot_product / (norm_a * norm_b)
}

fn main() {
    // load embedding model
    let mut model = TextEmbedding::try_new(
       InitOptions::new(EmbeddingModel::MxbaiEmbedLargeV1Q).with_show_download_progress(true),
    ).expect("Could not load model");

    // dummy input
    let documents = vec![
    "Hooks let your code \"do something\" when something happens in React - either the state changes, the lifecycle changes, or something else. ",
    "Binary crates are programs you can compile to an executable that you can run.",
    "A package is a bundle of one or more crates that provides a set of functionality. A package contains a Cargo.toml file that describes how to build those crates. ",
    // You can leave out the prefix but it's recommended
    "When declaring modules, you add an declaration statement, eg \"mod garden\" at the top of your crate root (src/lib.rs or src/main.rs) to say you're including this module. You can then add this mod's code in i) mod garden {} within the crate root, ii) in src/garden.rs, iii) or in src/garden/mod.rs.",
    ];

    // Generate embeddings with the default batch size, 256
    let doc_embeddings = model.embed(&documents, None).expect("Could not generate embeddings for documentation");

    // Raw user query
    let user_query = "Rules for declaring modules?";

    // Add relevant prompt to allow the user query to be used for retrieval
    let retrieval_query = format!{"Represent this sentence for searching relevant passages: {user_query}"};

    // Generate embedding from retrieval query
    let query_vector = [format!{"{retrieval_query}"}];

    let query_embedding = model.embed(query_vector, None).expect("Could not generate embeddings for query.");

    // Initialize variables to store best index and best similarity score
    let mut best_idx = 0;
    let mut best_score = 0.0;

    for (i, doc_emb) in doc_embeddings.iter().enumerate() {
        let score = cosine_similarity(&query_embedding[0], doc_emb);
        if score > best_score {
            best_score = score;
            best_idx = i;
        }
    }
    
    println!("Best match: {}", documents[best_idx]);
    println!("Similarity score: {:.4}", best_score);
}
