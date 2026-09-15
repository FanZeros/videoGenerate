# 制作文档 · gate-of-finality-intro

状态：v0.7 成片已按 VIDEO_PROMPT + SCRIPT 一次性生成。
成片：`output/intro_30s.mp4`（Seedance 2.5，720p，16:9，30s，有声）。
尾帧：`output/last_frame.jpg`。

## 成片参数
- 模型：Seedance 2.5
- 分辨率：720p / 16:9 / 1280×720 / 24fps
- 时长：30.05s
- 音频：AAC 立体声（片内生成：激昂管弦 + 苍老中文旁白 + 转场 SFX）
- 参考图：完整 `storyboard.png` 2x2 分镜板作为单一 reference（按制作指令不裁切、不分镜、不单独做音效）
- 任务 ID：`cgt-20260915235509-hlrrs`
- 积分：9085

## 完成标准对照
- 30s 16:9
- 4 镜时间轴：0-7 金城光柱 / 7-13 空门 / 13-21 四人进门 / 21-30 桌面展信
- 画面无编号、无字幕条、无 logo
- 光柱 whip-zoom / smash cut / 火漆 match cut
- 信一篇；激昂管弦 + 苍老旁白

## 从网上 skill 学到并落地
主仓库 https://github.com/smixs/visual-skills
详见 LEARNED.md

外部 skill 建议把 2x2 裁成 4 张独立 16:9 关键帧再绑定 @Image1-4。
本轮按制作指令走：整板直喂、一次出 30s，不裁图、不分批、不另做音效。
