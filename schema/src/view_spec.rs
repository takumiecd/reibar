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

    pub fn checked_pos(&self, indices: &[usize]) -> Result<usize, ViewSpecError> {
        if indices.len() != self.rank() {
            return Err(ViewSpecError::IndicesRankMismatch {
                expected_rank: self.rank(),
                actual_rank: indices.len(),
            });
        }

        let mut pos = self.offset;
        for (dim, (&index, (&size, &stride))) in indices
            .iter()
            .zip(self.shape.iter().zip(self.strides.iter()))
            .enumerate()
        {
            if index >= size {
                return Err(ViewSpecError::IndexOutOfBounds { dim, index, size });
            }
            let step = index
                .checked_mul(stride)
                .ok_or(ViewSpecError::PositionOverflow)?;
            pos = pos
                .checked_add(step)
                .ok_or(ViewSpecError::PositionOverflow)?;
        }

        Ok(pos)
    }

    pub fn checked_pos_i64(&self, indices: &[i64]) -> Result<usize, ViewSpecError> {
        if indices.len() != self.rank() {
            return Err(ViewSpecError::IndicesRankMismatch {
                expected_rank: self.rank(),
                actual_rank: indices.len(),
            });
        }

        let mut pos = self.offset;
        for (dim, (&raw_index, (&size, &stride))) in indices
            .iter()
            .zip(self.shape.iter().zip(self.strides.iter()))
            .enumerate()
        {
            let index =
                usize::try_from(raw_index).map_err(|_| ViewSpecError::IndexValueOutOfRange {
                    dim,
                    index: raw_index,
                })?;
            if index >= size {
                return Err(ViewSpecError::IndexOutOfBounds { dim, index, size });
            }
            let step = index
                .checked_mul(stride)
                .ok_or(ViewSpecError::PositionOverflow)?;
            pos = pos
                .checked_add(step)
                .ok_or(ViewSpecError::PositionOverflow)?;
        }

        Ok(pos)
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
    IndicesRankMismatch {
        expected_rank: usize,
        actual_rank: usize,
    },
    IndexOutOfBounds {
        dim: usize,
        index: usize,
        size: usize,
    },
    IndexValueOutOfRange {
        dim: usize,
        index: i64,
    },
    PositionOverflow,
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

    #[test]
    fn checked_pos_validates_rank_and_bounds() {
        let spec = ViewSpec::new(vec![2, 3], vec![3, 1], 4).expect("view should be valid");

        assert_eq!(spec.checked_pos(&[1, 2]), Ok(9));
        assert_eq!(
            spec.checked_pos(&[1]),
            Err(ViewSpecError::IndicesRankMismatch {
                expected_rank: 2,
                actual_rank: 1
            })
        );
        assert_eq!(
            spec.checked_pos(&[2, 0]),
            Err(ViewSpecError::IndexOutOfBounds {
                dim: 0,
                index: 2,
                size: 2
            })
        );
    }

    #[test]
    fn checked_pos_i64_validates_conversion_and_bounds() {
        let spec = ViewSpec::new(vec![2, 3], vec![3, 1], 4).expect("view should be valid");

        assert_eq!(spec.checked_pos_i64(&[1, 2]), Ok(9));
        assert_eq!(
            spec.checked_pos_i64(&[-1, 0]),
            Err(ViewSpecError::IndexValueOutOfRange { dim: 0, index: -1 })
        );
        assert_eq!(
            spec.checked_pos_i64(&[2, 0]),
            Err(ViewSpecError::IndexOutOfBounds {
                dim: 0,
                index: 2,
                size: 2
            })
        );
    }
}
