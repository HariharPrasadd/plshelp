use fastembed::{TextEmbedding, InitOptions, EmbeddingModel};
use std::any::type_name;
use rusqlite::{Connection, Result, params};
use std::fs;
use sentencex::segment;

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

fn clear_documents(conn: &Connection) -> Result<()> {
    conn.execute("DELETE FROM documents", [])?;
    Ok(())
}

// INPUT: Iterate over parallel vectors
fn insert_documents(
    conn: &Connection,
    contents: &Vec<&str>,
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

fn main() {
      // initialize connection to db
    let connection = init_db("plshelp.db").expect("Could not initialize database");

    // load embedding model
    let mut model = TextEmbedding::try_new(
       InitOptions::new(EmbeddingModel::MxbaiEmbedLargeV1Q).with_show_download_progress(true),
    ).expect("Could not load model");

    // dummy input string
    let docstring = fs::read_to_string("docs/acme.txt").expect("Could not load documentation file to string");
    
    // separate input string into sentences
    let documents: Vec<&str> = segment("en", &docstring);

    // Generate embeddings with the default batch size, 256
    let doc_embeddings: Vec<Vec<f32>> = model.embed(&documents, None).expect("Could not generate embeddings for documentation");

    // clear database before inserting new embeddings
    clear_documents(&connection).expect("Could not clear database");

    // Insert documents into sqlite db
    insert_documents(&connection, &documents, &doc_embeddings).expect("Could not add documents to database");

    // Raw user query
    let user_query = "Does AcmeStream guarantee exactly-once delivery?";

    // Generate query embedding by adding prefix and converting string to Vec<string>
    let query_embedding: Vec<Vec<f32>> = model.embed([format!{"Represent this sentence for searching relevant passages: {user_query}"}], None).expect("Could not generate embeddings for query.");

    // Results vector
    let results = search(&connection, &query_embedding, 1).expect("Could not search results.");

    // List all results in the format {index}. {answer} (score: {score})
    for (rank, (document, score)) in results.iter().enumerate() {
        println!("{}",  
            document.content,
        );
    }
}
