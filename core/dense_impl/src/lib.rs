#[derive(Debug, Clone, PartialEq)]
pub struct DenseTensorImpl {
    shape: Vec<usize>,
    data: Vec<f32>,
}

impl DenseTensorImpl {
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn data(&self) -> &[f32] {
        &self.data
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DenseBuilder {
    shape: Vec<usize>,
    data: Option<Vec<f32>>,
    fill_value: f32,
}

impl DenseBuilder {
    pub fn new(shape: Vec<usize>) -> Self {
        Self {
            shape,
            data: None,
            fill_value: 0.0,
        }
    }

    pub fn with_data(mut self, data: Vec<f32>) -> Self {
        self.data = Some(data);
        self
    }

    pub fn with_fill(mut self, value: f32) -> Self {
        self.fill_value = value;
        self
    }

    pub fn build(self) -> Result<DenseTensorImpl, DenseBuildError> {
        let expected = element_count(&self.shape)?;
        let data = match self.data {
            Some(data) => {
                if data.len() != expected {
                    return Err(DenseBuildError::DataLengthMismatch {
                        expected,
                        actual: data.len(),
                    });
                }
                data
            }
            None => vec![self.fill_value; expected],
        };

        Ok(DenseTensorImpl {
            shape: self.shape,
            data,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum DenseBuildError {
    DataLengthMismatch { expected: usize, actual: usize },
    ShapeOverflow,
}

fn element_count(shape: &[usize]) -> Result<usize, DenseBuildError> {
    shape.iter().try_fold(1usize, |acc, dim| {
        acc.checked_mul(*dim).ok_or(DenseBuildError::ShapeOverflow)
    })
}

#[cfg(test)]
mod tests {
    use super::{DenseBuildError, DenseBuilder};

    #[test]
    fn build_from_fill() {
        let tensor = DenseBuilder::new(vec![2, 3])
            .with_fill(1.5)
            .build()
            .expect("dense build should succeed");

        assert_eq!(tensor.shape(), &[2, 3]);
        assert_eq!(tensor.data(), &[1.5, 1.5, 1.5, 1.5, 1.5, 1.5]);
    }

    #[test]
    fn build_fails_on_data_len_mismatch() {
        let result = DenseBuilder::new(vec![2, 2])
            .with_data(vec![1.0, 2.0])
            .build();

        assert_eq!(
            result,
            Err(DenseBuildError::DataLengthMismatch {
                expected: 4,
                actual: 2,
            })
        );
    }
}
