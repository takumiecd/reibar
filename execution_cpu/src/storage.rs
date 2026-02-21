#[derive(Debug, Clone, PartialEq)]
pub struct CpuStorage {
    data: Vec<f32>,
}

impl CpuStorage {
    pub fn from_vec(data: Vec<f32>) -> Self {
        Self { data }
    }

    pub fn filled(len: usize, value: f32) -> Self {
        Self {
            data: vec![value; len],
        }
    }

    pub fn fill(&mut self, value: f32) {
        self.data.fill(value);
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }
}
