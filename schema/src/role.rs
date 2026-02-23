#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ArgRole {
    Input,
    Output,
    Temp,
    Param,
    Context,
}
