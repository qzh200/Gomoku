package main

import (
	"bufio"
	"io"
	"os"
	"path/filepath"
	"sync"
)

// prefixed 给引擎进程输出加 [名字] 前缀。
// 双击启动的窗口程序没有标准输出，此时写到 logs/<name>.log，排障时能拿到原因。
func prefixed(name string) io.Writer {
	target := io.Writer(os.Stdout)
	if !stdoutUsable() {
		if dir, err := executableDir(); err == nil {
			if err := os.MkdirAll(filepath.Join(dir, "logs"), 0o755); err == nil {
				file, err := os.OpenFile(filepath.Join(dir, "logs", name+".log"),
					os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o644)
				if err == nil {
					target = file
				}
			}
		}
	}
	p := &prefixWriter{name: "[" + name + "] ", out: bufio.NewWriter(target)}
	writersMu.Lock()
	writers = append(writers, p)
	writersMu.Unlock()
	return p
}

var (
	writersMu sync.Mutex
	writers   []*prefixWriter
)

// flushAll 在退出前把缓冲的输出写完，否则最后几行留在内存里。
func flushAll() {
	writersMu.Lock()
	defer writersMu.Unlock()
	for _, p := range writers {
		_ = p.out.Flush()
	}
}

type prefixWriter struct {
	name string
	out  *bufio.Writer
	mu   sync.Mutex
	head bool
}

func (p *prefixWriter) Write(data []byte) (int, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	written := 0
	for _, b := range data {
		if !p.head {
			if _, err := p.out.WriteString(p.name); err != nil {
				return written, err
			}
			p.head = true
		}
		if err := p.out.WriteByte(b); err != nil {
			return written, err
		}
		written++
		if b == '\n' {
			p.head = false
			if err := p.out.Flush(); err != nil {
				return written, err
			}
		}
	}
	return written, nil
}
