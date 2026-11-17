package main

import (
	"context"
	_ "embed"
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

const defaultAddr = "127.0.0.1:8080"

//go:embed assets/icon.png
var appIcon []byte

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

	runtimePath := os.Getenv(config.RuntimePathEnv)
	if runtimePath == "" {
		runtimePath = config.RuntimePath
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

	entries := make([]auth.EngineConfig, 0, len(cfg.Engines))
	for _, engine := range registry.Engines() {
		available := engine.Available
		entries = append(entries, auth.EngineConfig{
			EngineID:    engine.EngineID,
			DisplayName: engine.DisplayName,
			Token:       engine.Token,
			Available:   &available,
		})
	}
	if err := config.SaveRuntime(runtimePath, auth.Config{
		Engines:          entries,
		LocalClientToken: registry.LocalToken(),
	}); err != nil {
		log.Printf("写入运行时配置失败，Worker 需要手动配置令牌: %v", err)
	} else {
		log.Printf("引擎令牌已写入 %s", runtimePath)
	}

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
		if err := win.SetIcon(appIcon); err != nil {
			log.Printf("设置窗口图标失败: %v", err)
		}
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
