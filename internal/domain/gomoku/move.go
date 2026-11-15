package gomoku

// Move is one placed stone.
type Move struct {
	Row    int    `json:"row"`
	Col    int    `json:"col"`
	Player Player `json:"player"`
	Number int    `json:"number"`
}
