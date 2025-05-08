class InsufficientDataError(Exception):
    def __init__(self, n: int, required: int = 3):
        message = f"Only {n} samples provided, but at least {required} are required."
        super().__init__(message)
