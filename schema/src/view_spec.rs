#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ViewSpec {
    shape: Vec<usize>,
    strides: Vec<usize>,
    offset: usize,
}

impl ViewSpec {
    pub fn new(
        shape: Vec<usize>,
        strides: Vec<usize>,
        offset: usize,
    ) -> Result<Self, ViewSpecError> {
        if shape.len() != strides.len() {
            return Err(ViewSpecError::RankMismatch {
                shape_rank: shape.len(),
                strides_rank: strides.len(),
            });
        }
        Ok(Self {
            shape,
            strides,
            offset,
        })
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    pub fn offset(&self) -> usize {
        self.offset
    }

    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    pub fn checked_numel(&self) -> Result<usize, ViewSpecError> {
        self.shape
            .iter()
            .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
            .ok_or(ViewSpecError::NumelOverflow)
    }

    pub fn numel(&self) -> Option<usize> {
        self.checked_numel().ok()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ViewSpecError {
    RankMismatch {
        shape_rank: usize,
        strides_rank: usize,
    },
    NumelOverflow,
}

#[cfg(test)]
mod tests {
    use super::{ViewSpec, ViewSpecError};

    #[test]
    fn new_rejects_rank_mismatch() {
        let err = ViewSpec::new(vec![2, 3], vec![3], 0).expect_err("rank mismatch should fail");
        assert_eq!(
            err,
            ViewSpecError::RankMismatch {
                shape_rank: 2,
                strides_rank: 1
            }
        );
    }

    #[test]
    fn checked_numel_detects_overflow() {
        let spec = ViewSpec::new(vec![usize::MAX, 2], vec![1, 1], 0)
            .expect("rank-matched shape/strides should construct");
        assert_eq!(spec.checked_numel(), Err(ViewSpecError::NumelOverflow));
    }
}
