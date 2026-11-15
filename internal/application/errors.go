package application

import "fmt"

// Error is an application level failure. The transport layer maps its code to
// an HTTP status.
type Error struct {
	Code    string
	Message string
}

func (e *Error) Error() string { return e.Code + ": " + e.Message }

// The error codes of the frozen API contract.
const (
	CodeInvalidRequest     = "INVALID_REQUEST"
	CodeUnauthorized       = "UNAUTHORIZED"
	CodeForbidden          = "FORBIDDEN"
	CodeNotYourTurn        = "NOT_YOUR_TURN"
	CodeGameNotFound       = "GAME_NOT_FOUND"
	CodeClaimNotFound      = "CLAIM_NOT_FOUND"
	CodeVersionConflict    = "VERSION_CONFLICT"
	CodePositionOccupied   = "POSITION_OCCUPIED"
	CodeGameFinished       = "GAME_FINISHED"
	CodeGamePaused         = "GAME_PAUSED"
	CodeClaimExpired       = "CLAIM_EXPIRED"
	CodeClaimInvalid       = "CLAIM_INVALID"
	CodeIdempotencyReused  = "IDEMPOTENCY_KEY_REUSED"
	CodeInvalidGameConfig  = "INVALID_GAME_CONFIG"
	CodePositionOutOfRange = "POSITION_OUT_OF_RANGE"
	CodeInternalError      = "INTERNAL_ERROR"
)

// newError builds an application error.
func newError(code, message string) *Error {
	return &Error{Code: code, Message: message}
}

// newErrorf builds an application error with a formatted message.
func newErrorf(code, format string, args ...any) *Error {
	return &Error{Code: code, Message: fmt.Sprintf(format, args...)}
}
