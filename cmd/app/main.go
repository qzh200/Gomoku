// Command app runs the MyGo desktop application: the RESTful API for the
// Python AI workers and the native desktop interface.
package main

import (
	"context"
	"errors"
	"fmt"
	"log"
	"net"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"syscall"
	"time"

	"github.com/egoist/mygo"
	mygoui "github.com/egoist/mygo/ui"

	"mygo/internal/application"
	"mygo/internal/auth"
	"mygo/internal/config"
	"mygo/internal/repository"
	"mygo/internal/scheduler"
	httpapi "mygo/internal/transport/http"
	"mygo/ui"
)

// defaultAddr is the address of the local API. It binds the loopback
// interface only, the service is never exposed to the network by default.
const defaultAddr = "127.0.0.1:8080"

func main() {
	log.SetFlags(0)
	log.SetPrefix("mygo: ")

	addr := os.Getenv("MYGO_ADDR")
	if addr == "" {
		addr = defaultAddr
	}
	if err := run(addr); err != nil {
		log.Fatal(err)
	}
}

// run starts the API server and the desktop window.
func run(addr string) error {
	configPath := os.Getenv(config.PathEnv)
	if configPath == "" {
		configPath = config.Path
	}
	cfg, found, err := config.Load(configPath)
	if err != nil {
		return err
	}
	registry, err := auth.NewRegistry(cfg)
	if err != nil {
		return err
	}
	if !found {
		log.Printf("未找到 %s，使用内置引擎与随机令牌", configPath)
	}

	games := repository.NewMemoryGames()
	leases := scheduler.NewLeaseManager()
	idem := repository.NewMemoryIdempotency()
	coord := application.NewCoordinator(games, leases, idem)

	gameService := application.NewGameService(coord, registry)
	moveService := application.NewMoveService(coord)
	claimService := application.NewClaimService(coord)

	server := httpapi.NewServer(gameService, moveService, claimService, registry)
	listener, err := net.Listen("tcp", addr)
	if err != nil {
		return fmt.Errorf("监听 %s 失败: %w", addr, err)
	}
	httpServer := &http.Server{
		Handler:           server.Handler(),
		ReadHeaderTimeout: 10 * time.Second,
	}
	go func() {
		if err := httpServer.Serve(listener); err != nil && !errors.Is(err, http.ErrServerClosed) {
			log.Printf("HTTP 服务结束: %v", err)
		}
	}()
	defer func() {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		defer cancel()
		_ = httpServer.Shutdown(ctx)
	}()

	log.Printf("API 已启动: http://%s/api/v1", listener.Addr().String())
	log.Printf("本地客户端令牌: %s", registry.LocalToken())
	for _, engine := range registry.Engines() {
		line := fmt.Sprintf("引擎 %s (%s) 可用=%v", engine.EngineID, engine.DisplayName, engine.Available)
		if engine.Generated {
			line += " token=" + engine.Token
		}
		log.Print(line)
	}

	app := ui.New(gameService, moveService, registry)
	mygo.App.WhenReady(func() {
		win := mygo.NewWindow(mygo.WindowOptions{
			Title:           "五子棋",
			Width:           1180,
			Height:          800,
			MinWidth:        960,
			MinHeight:       640,
			BackgroundColor: "light-dark(#f5f5f5, #17181c)",
			Content:         mygoui.View(app.View()),
		})
		app.AttachWindow(win)
		app.StartPolling()
	})

	stop := make(chan os.Signal, 1)
	signal.Notify(stop, os.Interrupt, syscall.SIGTERM)
	go func() {
		<-stop
		mygo.App.Quit()
	}()

	if err := mygo.App.Run(); err != nil {
		return err
	}
	return nil
}

// portOf returns the port of an address.
func portOf(addr string) int {
	_, port, err := net.SplitHostPort(addr)
	if err != nil {
		return 0
	}
	value, err := strconv.Atoi(port)
	if err != nil {
		return 0
	}
	return value
}
