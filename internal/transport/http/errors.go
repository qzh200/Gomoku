package http

import (
	"encoding/json"
	"errors"
	"net/http"

	"mygo/internal/application"
)

type errorBody struct {
	Error errorDetail `json:"error"`
}

type errorDetail struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

var statusFor = map[string]int{
	application.CodeInvalidRequest:     http.StatusBadRequest,
	application.CodeUnauthorized:       http.StatusUnauthorized,
	application.CodeForbidden:          http.StatusForbidden,
	application.CodeNotYourTurn:        http.StatusForbidden,
	application.CodeGameNotFound:       http.StatusNotFound,
	application.CodeClaimNotFound:      http.StatusNotFound,
	application.CodeVersionConflict:    http.StatusConflict,
	application.CodePositionOccupied:   http.StatusConflict,
	application.CodeGameFinished:       http.StatusConflict,
	application.CodeGamePaused:         http.StatusConflict,
	application.CodeClaimExpired:       http.StatusConflict,
	application.CodeClaimInvalid:       http.StatusConflict,
	application.CodeIdempotencyReused:  http.StatusConflict,
	application.CodeInvalidGameConfig:  http.StatusUnprocessableEntity,
	application.CodePositionOutOfRange: http.StatusUnprocessableEntity,
	application.CodeInternalError:      http.StatusInternalServerError,
}

func writeError(w http.ResponseWriter, err error) {
	var appErr *application.Error
	if !errors.As(err, &appErr) {
		appErr = &application.Error{Code: application.CodeInternalError, Message: err.Error()}
	}
	status, ok := statusFor[appErr.Code]
	if !ok {
		status = http.StatusInternalServerError
	}
	writeJSON(w, status, errorBody{Error: errorDetail{Code: appErr.Code, Message: appErr.Message}})
}

func writeJSON(w http.ResponseWriter, status int, payload any) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(status)
	if payload == nil {
		return
	}
	_ = json.NewEncoder(w).Encode(payload)
}
