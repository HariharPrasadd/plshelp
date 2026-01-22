use fastembed::{TextEmbedding, InitOptions, EmbeddingModel};
use std::any::type_name;
use rusqlite::{Connection, Result, params};

/* mixedbread model information

   ModelInfo {
        model: MxbaiEmbedLargeV1Q,
        dim: 1024,
        description: "Quantized Large English embedding model from MixedBread.ai",
        model_code: "mixedbread-ai/mxbai-embed-large-v1",
        model_file: "onnx/model_quantized.onnx",
        additional_files: [],
        output_key: None,
    },
*/

// struct for structured output
#[derive(Debug)]
struct Document {
    id: i64,
    content: String,
    embedding: Vec<f32>,
}

// helper function to determine type of input
fn type_of<T>(_: T) -> &'static str {
    type_name::<T>()
}

// Helper functions for embedding conversion
fn embedding_to_bytes(embedding: &[f32]) -> Vec<u8> {
    embedding
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect()
}

fn bytes_to_embedding(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

/// Compute cosine similarity between two vectors
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    
    dot_product / (norm_a * norm_b)
}

// initialize database with appropriate structure
fn init_db(db_path: &str) -> Result<Connection> {
    let conn = Connection::open(db_path)?;
    
    conn.execute(
        "CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            embedding BLOB NOT NULL
        )",
        [],
    )?;
    
    Ok(conn)
}

// INPUT: Iterate over parallel vectors
fn insert_documents(
    conn: &Connection,
    contents: &Vec<String>,
    embeddings: &[Vec<f32>],
) -> Result<()> {
    assert_eq!(contents.len(), embeddings.len());
    
    for i in 0..contents.len() {
        let embedding_bytes = embedding_to_bytes(&embeddings[i]);
        
        conn.execute(
            "INSERT INTO documents (content, embedding) VALUES (?1, ?2)",
            params![contents[i], embedding_bytes],
        )?;
    }
    
    Ok(())
}

// OUTPUT: Search with natural language query
fn search(
    connection: &Connection,
    query_embedding: &Vec<Vec<f32>>,
    top_k: usize,
) -> Result<Vec<(Document, f32)>> {
    let mut statement = connection.prepare(
        "SELECT id, content, embedding FROM documents"
    )?;
    
    let mut results: Vec<(Document, f32)> = statement
        .query_map([], |row| {
            let document_id: i64 = row.get(0)?;
            let document_content: String = row.get(1)?;
            let embedding_bytes: Vec<u8> = row.get(2)?;
            
            let document_embedding = bytes_to_embedding(&embedding_bytes);
            let similarity_score = cosine_similarity(&query_embedding[0], &document_embedding);
            
            let document = Document {
                id: document_id,
                content: document_content,
                embedding: document_embedding,
            };
            
            Ok((document, similarity_score))
        })?
        .collect::<Result<Vec<_>>>()?;
    
    // Sort by score descending
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    
    // Take top k
    results.truncate(top_k);
    
    Ok(results)
}

/// Semantic chunking function that splits text based on semantic similarity
/// Returns a vector of text chunks where boundaries are determined by drops in similarity
fn semantic_chunk(
    text: &str,
    model: &mut TextEmbedding,
    similarity_threshold: f32,  // e.g., 0.75 - chunks split when similarity drops below this
) -> Vec<String> {
    // 1. Split text into sentences (simple version - splits on period, exclamation, question mark)
    let sentences: Vec<String> = text
        .split(|c| c == '.' || c == '!' || c == '?')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty() && s.len() > 3)  // Filter out very short fragments
        .collect();
    
    // Edge case: if only one sentence, return it as single chunk
    if sentences.len() <= 1 {
        return vec![text.to_string()];
    }
    
    // 2. Generate embeddings for each sentence
    let sentence_refs: Vec<&str> = sentences.iter().map(|s| s.as_str()).collect();
    let embeddings: Vec<Vec<f32>> = model
        .embed(sentence_refs, None)
        .expect("Could not generate sentence embeddings");
    
    // 3. Find chunk boundaries based on cosine similarity drops
    let mut chunks: Vec<String> = Vec::new();
    let mut chunk_start = 0;
    
    // CHUNKING LOGIC:
    // - Compare each consecutive sentence pair (i, i+1) for semantic similarity
    // - If similarity is HIGH: keep chunk_start where it is (sentences stay together)
    // - If similarity is LOW (< threshold): create a boundary
    //   → This captures ALL sentences from chunk_start to i (could be many sentences!)
    //   → Then move chunk_start to i+1 to start a new chunk
    // 
    // Example: [S0, S1, S2, S3] with threshold=0.75
    //   i=0: S0↔S1 similarity=0.85 → NO boundary (chunk_start stays 0)
    //   i=1: S1↔S2 similarity=0.82 → NO boundary (chunk_start still 0)
    //   i=2: S2↔S3 similarity=0.60 → BOUNDARY! 
    //        → Create chunk from sentences[0..=2] = [S0, S1, S2] (3 sentences!)
    //        → chunk_start = 3
    //
    // This naturally groups multiple consecutive similar sentences together
    // by keeping chunk_start fixed until semantic similarity drops.
    for i in 0..embeddings.len() - 1 {
        let similarity = cosine_similarity(&embeddings[i], &embeddings[i + 1]);
        
        // If similarity drops below threshold, create a chunk boundary
        if similarity < similarity_threshold {
            // Combine sentences from chunk_start to i (inclusive)
            let chunk_text = sentences[chunk_start..=i].join(". ") + ".";
            chunks.push(chunk_text);
            chunk_start = i + 1;
        }
    }
    
    // Add the final chunk (remaining sentences that weren't chunked in the loop)
    // This happens when the last few sentences were all similar to each other,
    // so no boundary was created and they're still waiting to be added.
    if chunk_start < sentences.len() {
        let chunk_text = sentences[chunk_start..].join(". ") + ".";
        chunks.push(chunk_text);
    }
    
    chunks
}

fn main() {
    // initialize connection to db
    let connection = init_db("plshelp.db").expect("Could not initialize database");

    // load embedding model
    let mut model = TextEmbedding::try_new(
       InitOptions::new(EmbeddingModel::MxbaiEmbedLargeV1Q).with_show_download_progress(true),
    ).expect("Could not load model");

    // dummy input
    let _documents_old = vec![
    "A closure in Rust is an anonymous function that can capture variables from its surrounding scope, allowing you to pass behavior as a value.",
    "In HTTP, a 404 status code means the server was reached successfully but the requested resource could not be found.",
    "A hash map stores key–value pairs and provides average constant-time lookup by computing a hash of the key.",
    "In machine learning, overfitting occurs when a model learns noise in the training data and performs poorly on unseen examples.",
    "Garbage collection is a form of automatic memory management where the runtime periodically reclaims memory that is no longer reachable by the program.",
    ];

    // dummy input string
    let docstring = "A closure in Rust is an anonymous function that can capture variables from its surrounding scope, allowing you to pass behavior as a value. In HTTP, a 404 status code means the server was reached successfully but the requested resource could not be found. A hash map stores key–value pairs and provides average constant-time lookup by computing a hash of the key. In machine learning, overfitting occurs when a model learns noise in the training data and performs poorly on unseen examples. Garbage collection is a form of automatic memory management where the runtime periodically reclaims memory that is no longer reachable by the program.";
    let documents = semantic_chunk(docstring, &mut model, 0.75);

    for doc in &documents {
        println!("{}", doc);
    }

    // Generate embeddings with the default batch size, 256
    let doc_embeddings: Vec<Vec<f32>> = model.embed(&documents, None).expect("Could not generate embeddings for documentation");

    // Insert documents into sqlite db
    insert_documents(&connection, &documents, &doc_embeddings).expect("Could not add documents to database.");

    // Raw user query
    let user_query = "Why does my model not accurately predict out of distribution values?";

    // Generate query embedding by adding prefix and converting string to Vec<string>
    let query_embedding: Vec<Vec<f32>> = model.embed([format!{"Represent this sentence for searching relevant passages: {user_query}"}], None).expect("Could not generate embeddings for query.");

    // Results vector
    let results = search(&connection, &query_embedding, 3).expect("Could not search results.");

    // List all results in the format {index}. {answer} (score: {score})
    for (rank, (document, score)) in results.iter().enumerate() {
        println!("{}. {} (score: {:.4})", 
            rank + 1, 
            document.content,
            score,
        );
    }
}
