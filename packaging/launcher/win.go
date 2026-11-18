package main

import (
	"fmt"
	"os"
	"path/filepath"
	"syscall"
	"unsafe"
)

// executableDir 返回 exe 所在目录，不解析软链接。
func executableDir() (string, error) {
	path, err := os.Executable()
	if err != nil {
		return "", err
	}
	return filepath.Dir(path), nil
}

var (
	kernel32     = syscall.NewLazyDLL("kernel32.dll")
	getStdHandle = kernel32.NewProc("GetStdHandle")
	messageBox   = syscall.NewLazyDLL("user32.dll").NewProc("MessageBoxW")
)

// stdoutUsable 判断进程是否有可写的标准输出句柄。
// 只看句柄是否有效，不看它是不是控制台：重定向到文件/管道时也必须能写。
func stdoutUsable() bool {
	const invalid = ^uintptr(0)
	handle, _, _ := getStdHandle.Call(uintptr(^uint32(0) - 10)) // STD_OUTPUT_HANDLE
	return handle != 0 && handle != invalid
}

// fatal 打印错误；无控制台时弹窗，否则双击运行时错误不可见。
func fatal(format string, args ...any) {
	message := fmt.Sprintf(format, args...)
	if !stdoutUsable() {
		title, _ := syscall.UTF16PtrFromString("MyGo 启动失败")
		body, _ := syscall.UTF16PtrFromString(message)
		_, _, _ = messageBox.Call(0, uintptr(unsafe.Pointer(body)), uintptr(unsafe.Pointer(title)), 0x10)
		return
	}
	flushAll()
	_, _ = os.Stderr.WriteString(message + "\n")
	os.Exit(1)
}
