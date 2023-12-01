# Sconce

Example training code

```rust
fn train_example() -> Result<()> {
    let device = &Device::Cpu;
    let variables = VarMap::new();
    let model = Sequential::new()
        .add_layer(&Dense::new(2).kernel_initializer(Some(ZERO)).build())
        .compile(&variables, &Device::Cpu, nll, Optimizers::AdamWDefault)?;
    let x = &Tensor::new(&[1.], &device)?;
    let y = &Tensor::new(&[1.], &device)?;

    let output = model.fit(x, y, 10)?;

    println!("Loss: {}", output.loss);

    Ok(())
}
```