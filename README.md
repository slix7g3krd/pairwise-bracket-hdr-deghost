# pairwise-bracket-hdr-deghost

## 领域

游戏、图形与多媒体

## 主题

包围曝光 HDR 去鬼影

## 说明

局部对齐、亮度差掩码与解码屏障。

初始状态下 `imaging/bracket_hdr_deghost.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
