#![no_std]
#![no_main]

use burn::backend::ndarray::NdArray;
use burn::tensor;
use esp_backtrace as _;
use esp_hal::{
    clock::ClockControl, delay::Delay, peripherals::Peripherals, prelude::*, system::SystemControl,
};
use espae::mnist::Model;

type Backend = NdArray<f32>;

#[entry]
fn main() -> ! {
    let peripherals = Peripherals::take();
    let system = SystemControl::new(peripherals.SYSTEM);

    let clocks = ClockControl::max(system.clock_control).freeze();
    let delay = Delay::new(&clocks);

    esp_println::logger::init_logger_from_env();

    let device = Default::default();

    let model: Model<Backend> = Model::new(&device);

    let input = tensor::Tensor::<NdArray<f32>, 4>::zeros([1, 1, 28, 28], &device);

    let output = model.forward(input);

    log::info!("{:?}", output);

    loop {
        log::info!("Hello world!");
        delay.delay(500.millis());
    }
}
