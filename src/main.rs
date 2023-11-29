use anyhow::Result;
use qdrant_client::prelude::*;
use qdrant_client::qdrant::vectors_config::Config;
use qdrant_client::qdrant::{
    Condition, CreateCollection, Filter, SearchPoints, VectorParams, VectorsConfig,
};
use rust_bert::pipelines::sentence_embeddings::SentenceEmbeddingsBuilder;
use serde_json::json;

#[tokio::main]
async fn main() -> Result<()> {
    let model =
        SentenceEmbeddingsBuilder::local("/home/zhang/Desktop/all-MiniLM-L6-v2/").create_model()?;
    // Example of top level client
    // You may also use tonic-generated client from `src/qdrant.rs`
    let client = QdrantClient::from_url("http://192.168.237.1:6334").build()?;

    let collections_list = client.list_collections().await?;
    // dbg!(collections_list);
    // collections_list = ListCollectionsResponse {
    //     collections: [
    //         CollectionDescription {
    //             name: "test",
    //         },
    //     ],
    //     time: 1.78e-6,
    // }

    // Define input
    let sentences = ["这是一句话", "every sentence get converted to vector"];

    // Generate Embeddings
    let embeddings = model.encode(&sentences)?;
    dbg!(embeddings[1].len());
    // println!("{embeddings:?}");

    let collection_name = "test";
    client.delete_collection(collection_name).await?;

    client
        .create_collection(&CreateCollection {
            collection_name: collection_name.into(),
            vectors_config: Some(VectorsConfig {
                config: Some(Config::Params(VectorParams {
                    size: 384,
                    distance: Distance::Cosine.into(),
                    ..Default::default()
                })),
            }),
            ..Default::default()
        })
        .await?;

    let collection_info = client.collection_info(collection_name).await?;
    // dbg!(collection_info);

    let payload: Payload = json!(
        {
            "foo": "Bar",
            "bar": 12,
            "baz": {
                "qux": "quux"
            }
        }
    )
    .try_into()
    .unwrap();

    let points = vec![PointStruct::new(0, embeddings[1].to_vec(), payload)];
    client
        .upsert_points_blocking(collection_name, points, None)
        .await?;

    let search_result = client
        .search_points(&SearchPoints {
            collection_name: collection_name.into(),
            vector: vec![0.04374401; 384],
            filter: Some(Filter::all([Condition::matches("bar", 12)])),
            limit: 10,
            with_payload: Some(true.into()),
            ..Default::default()
        })
        .await?;
    // dbg!(&search_result);
    // search_result = SearchResponse {
    //     result: [
    //         ScoredPoint {
    //             id: Some(
    //                 PointId {
    //                     point_id_options: Some(
    //                         Num(
    //                             0,
    //                         ),
    //                     ),
    //                 },
    //             ),
    //             payload: {
    //                 "bar": Value {
    //                     kind: Some(
    //                         IntegerValue(
    //                     12,
    //                     ),
    //                     ),
    //                 },
    //                 "foo": Value {
    //                     kind: Some(
    //                         StringValue(
    //                     "Bar",
    //                     ),
    //                     ),
    //                 },
    //             },
    //             score: 1.0000001,
    //             version: 0,
    //             vectors: None,
    //         },
    //     ],
    //     time: 9.5394e-5,
    // }

    let found_point = search_result.result.into_iter().next().unwrap();
    let mut payload = found_point.payload;
    let baz_payload = payload.remove("baz").unwrap().into_json();
    println!("baz: {}", baz_payload);
    // baz: {"qux":"quux"}

    Ok(())
}
