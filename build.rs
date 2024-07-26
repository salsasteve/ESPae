use burn_import::onnx::{ModelGen, RecordType};

fn main() {
    ModelGen::new()
        .input("src/model/mnist.onnx")
        .out_dir("model/")
        .record_type(RecordType::Bincode)
        .embed_states(true)
        .run_from_script();
}
