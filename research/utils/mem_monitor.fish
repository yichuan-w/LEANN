#!/usr/bin/env fish

# 用法: ./mem_monitor.fish <PID> [interval_seconds=5]
# 比如： ./mem_monitor.fish 110303 5
# 会在当前目录写一个 mem_usage_110303.log，每隔5秒记录一次RSS和VSZ（单位MB）。

function usage
    echo "用法: mem_monitor.fish <PID> [interval_seconds=5]"
    exit 1
end

if test (count $argv) -lt 1
    usage
end

set pid $argv[1]
set interval 5
if test (count $argv) -gt 1
    set interval $argv[2]
end

# 输出到 mem_usage_<pid>.log
set logfile mem_usage_$pid.log
echo "写入日志: $logfile"
echo "timestamp,rss_MB,vms_MB" > $logfile

# 轮询检查
while true
    # 获取 RSS/VSZ (KB) 值
    # 兼容 macOS 的 ps 命令，不使用 Linux 特有的选项
    set proc_line (ps -p $pid -o rss,vsz | tail -n +2 2>/dev/null)

    # 若取不到(进程已退出)，则停止
    if test -z "$proc_line"
        echo "进程 $pid 已退出或不存在."
        exit 0
    end

    # 将单行字符串通过空格拆分为数组，如 ("79673856" "95904664")
    set arr (string split ' ' (string trim $proc_line))
    if test (count $arr) -lt 2
        echo "解析 ps 输出时出现意外: $proc_line"
        exit 1
    end

    # 分别赋值
    set rss_kb $arr[1]
    set vsz_kb $arr[2]

    # 时间戳
    set t (date "+%Y-%m-%d %H:%M:%S")

    # 转换成 MB
    set rss_MB (math "$rss_kb / 1024.0")
    set vsz_MB (math "$vsz_kb / 1024.0")

    # 写日志
    echo "$t,$rss_MB,$vsz_MB" >> $logfile

    sleep $interval
end
