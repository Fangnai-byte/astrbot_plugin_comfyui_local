# ComfyUI 本地出图（AstrBot 插件）

![logo](logo.png)

把**你本机的 ComfyUI** 接进 AstrBot：群里一句话出图，模型和显存都是你自己的，
不走任何第三方 API。

* 直接吃 ComfyUI 的**界面格式**工作流 JSON（菜单里普通「导出」那种），提交前自动转
  API 格式 —— 不用手动再导一遍 API。
* 出图走 **WebSocket** 拿结果（比轮询 `/history` 可靠，`/history` 在 ComfyUI 的
  SQLite 起不来时会永远是空的）。
* 文生图 / 图生图 / ControlNet 锁结构，都支持。
* 带**内容兜底**（提示词 + 产出图两道门）、**出图频率限制**、**出图日志**和
  **人设台词**。

> ⚠️ **本发行版不带示例工作流。** 你需要先从自己的 ComfyUI 导出一份（见下面
> 「第二步」）。没有工作流插件不会出图。

---

## 下载

* **Release（推荐）**：到 [Releases](https://github.com/Fangnai-byte/astrbot_plugin_comfyui_local/releases)
  下载 `astrbot_plugin_comfyui_local.zip`，解压后把整个目录放进插件目录。
* **git**：

  ```
  git clone https://github.com/Fangnai-byte/astrbot_plugin_comfyui_local.git
  ```

* **更新**：`git pull`，或用新 zip 覆盖旧目录。你改过的配置不会丢 ——
  真正生效的工作流 / 预设 / 参考图都在 `plugin_data/` 里，不在插件目录。
* **问题与建议**：<https://github.com/Fangnai-byte/astrbot_plugin_comfyui_local/issues>

---

## 一、安装

1. 需要 **AstrBot 4.16 ~ 4.28**（`astrbot_version: >=4.16,<5`）。
2. 把这个目录整个放进 AstrBot 的插件目录：

   ```
   <AstrBot>/data/plugins/astrbot_plugin_comfyui_local/
   ```

3. 装依赖（只有一个）：

   ```
   pip install aiohttp>=3.9
   ```

   如果你用的是 AstrBot 自带的虚拟环境：
   `<AstrBot>/.venv/Scripts/python.exe -m pip install aiohttp`
   （Linux/macOS 是 `.venv/bin/python -m pip install aiohttp`）

4. 重启 AstrBot。日志里应出现：

   ```
   Loading plugin astrbot_plugin_comfyui_local ...
   [comfyui] 载入 0 个工作流        ← 还没有工作流，正常
   ```

5. 在 AstrBot 的 WebUI 里打开这个插件的配置，确认 `server` 指向你的 ComfyUI
   （默认 `http://127.0.0.1:8188`）。

---

## 二、第二步：从 ComfyUI 导出工作流（必做）

插件**只认 ComfyUI 网页里跑通过的工作流**，所以先在你自己的 ComfyUI 里调好一张图：

1. 浏览器打开 ComfyUI，加载/搭好一个能出图的工作流，**实际跑通一次**。
2. 菜单 → **工作流 / Workflow → 导出（Export）**，存成 `.json`。
   这个就是所谓的「界面格式」（文件里有 `nodes`、`links`）。
   *不用* 去用「导出（API）」那个 —— 插件会自己转。
3. 把 json 放进插件的**数据目录**（不是插件目录）：

   ```
   <AstrBot>/data/plugin_data/astrbot_plugin_comfyui_local/workflows/
   ```

   > 插件目录下也有个 `workflows/`，那是**首次运行的播种模板**。实际生效的是
   > `plugin_data` 里这个。想放在别处，就在配置里加 `extra_workflow_dirs`。

4. 在聊天里发 `/画图 重载`（不用重启），然后 `/画图 工作流` 应该能看到它。

5. 试一张：

   ```
   /画图 一只白猫在窗台上，午后阳光
   ```

**关于提示词语言（重要）**：底层是 SDXL/Illustrious 系模型，只认**英文 danbooru
标签**。中文有一些常用说法内置了翻译表（姿势、表情、颜色、服装等），但生僻说法
会被忽略，聊天里会提醒你哪些词没翻出来。想稳定就写英文标签，例如：

```
/画图 1girl, solo, white hair, kimono, sitting, reading book
```

---

## 三、常用命令

| 命令 | 作用 |
|---|---|
| `/画图 描述` | 直接出图 |
| `/画图 -w 工作流名 描述` | 指定工作流 |
| `/画图 -p 预设名 描述` | 套预设（角色/风格），描述只写要追加的部分 |
| `/画图 -r 参考图名 描述` | 图生图：拿内置参考图当底图 |
| `/画图 -r 参考图名 -c 描述` | ControlNet：只抄参考图的结构/姿势，内容重画 |
| `/画图 -r 参考图名 -d 0.7 描述` | 调重绘幅度（默认 0.55，越小越像参考图） |
| `/画图 -n 描述` | 这一张不套预设 |
| `/画图 工作流` | 列出工作流 |
| `/画图 预设` | 列出预设 |
| `/画图 参考图` | 列出参考图 |
| `/画图 日志` | 看最近的出图记录（时间/预设/参数/耗时/暴露度） |
| `/画图 人设` | 看回执台词 |
| `/画图 状态` | 检查 ComfyUI 连通性 |
| `/画图 重载` | 重新扫描工作流、预设、参考图、台词 |

**自然语言入口**：插件也注册了一个 LLM 工具（`draw_image`）。挂上模型后，对方说
「画个……」机器人会自己调用；提示词由模型翻译成英文标签。命令和自然语言**共存**。

---

## 四、预设与参考图

### 预设

预设 = 一套调好的出图配置（基础提示词 + 参数 + 可选参考图）。
文件在 `<AstrBot>/data/plugin_data/astrbot_plugin_comfyui_local/presets.json`，
**默认是空的**（只有注释和一份 `_example_` 示例格式）。

关键规则：**预设的 `base_prompt` 会拼在你的描述前面**，用户的输入只做追加 ——
这样 LoRA 的触发词不会被顶掉。所以预设里那串角色 tag 是必须的。

`base_prompt` 也可以写成 `base_prompt_file` 指向一个 txt
（例如你 ComfyUI 目录里已经写好的提示词文件），省得抄两遍。

### 参考图（图生图 / ControlNet）

把 PNG 放进 `<AstrBot>/data/plugin_data/astrbot_plugin_comfyui_local/refs/`，
文件名（不含扩展名）就是 `-r` 后面写的名字。透明背景会自动合成到白底。

用参考图时**姿势和背景会被锁住** —— 这是 img2img 的固有行为，不是 bug。
所以：

* 想**换姿势或换场景**（坐着、看窗边…）就**别写 `-r`**，插件也会自动帮你去掉
  预设绑定的参考图，让文字自由构图；
* 想**保留姿势只换表情/细节**才用 `-r`；
* 想**换衣服但姿势不变**用 `-r 参考图 -c 描述`（ControlNet）。

> 代价要说清楚：去掉参考图后，**官方配色也会跟着漂**（参考图同时锁着颜色）。
> 二者不可兼得。

---

## 五、内容兜底、频率限制、日志

### 内容兜底：两道门

| 门 | 时机 | 可靠性 | 行为 |
|---|---|---|---|
| **提示词门** | 生成**之前** | ✅ 精确 | 描述里有露骨/不当关键词时直接拒绝，不浪费显卡 |
| **产出图门** | 生成之后 | ⚠️ 启发式 | 按像素估「暴露度」，超阈值不发，图挪到 `blocked/` |

> ⚠️ **产出图门是兜底，不是保证。** 它只用 PIL + numpy 算「皮肤面积」这类像素
> 特征，**会把类别排错**（实测：开领较深的正常图可能比裸体图分值还高，比基尼又
> 可能比裸体低），所以既会误拦也会漏放。默认阈值 55（= 0.55，越低越严）。
> 每张图的实测分值都写进日志（`exposure=0.10@0.40-0.46`），照着日志调
> `nsfw_skin_threshold` 即可。要真正可靠得换训练过的分类器。

**谁能收到「可能露骨」的产出**：**群聊一律不发**；私聊要同时满足
`enable_r18` 打开 **且** 在 `r18_user_whitelist` 里。被拦的图不删，挪到 `blocked/`。

> 这两个键是**权限门，不是素材开关**。本发行版不附带任何露骨素材，
> 打开它只意味着「白名单用户私聊时不被内容门拦」，插件并不会因此多出什么内容；
> 关掉它则是连白名单也一起拦，等于最严模式。

### 频率限制

按**发送者 ID** 计数（换群共用配额，换群刷没用）：

* `rate_limit_cooldown`（默认 10 秒）—— 挡手抖连点
* `rate_limit_window` / `rate_limit_count`（默认 600 秒 5 张）—— 挡一口气刷十张

### 出图日志

`<AstrBot>/data/plugin_data/astrbot_plugin_comfyui_local/draw.log`，一行一条：

```
2026-01-01 12:00:00 | ok | user=1000 | chat=group:123 | req=... | preset=... | wf=...
  | ref=... | mode=img2img | denoise=0.45 | size=832x1216 | seed=123456789
  | elapsed=12.3s | out=xxx.png | exposure=0.12@0.40-0.46 | note=...
```

`seed` 记下来了，任何一张都能事后复现。文件超过 `draw_log_max_kb` 会轮转成
`draw.log.1`。`/画图 日志` 在聊天里看最近几条。

### 人设台词

默认回执是一句有人味的台词（按场景分组随机取），而不是机器腔。台词文件在
`<AstrBot>/data/plugin_data/astrbot_plugin_comfyui_local/persona.json`，
首次运行自动生成，直接改，改完 `/画图 重载`。想换成朴素文案就把
`persona_enable` 关掉。

---

## 六、配置项

WebUI 里能改的都在 `_conf_schema.json` 里，常用的：

| 配置 | 默认 | 说明 |
|---|---|---|
| `server` | `http://127.0.0.1:8188` | ComfyUI 地址 |
| `command` | `画图` | 主指令名 |
| `default_workflow` | 空 | 默认工作流；留空则用唯一可用的那个 |
| `default_img2img_workflow` | 空 | 图生图工作流，留空自动挑含 LoadImage 的 |
| `img2img_denoise` | `55` | 图生图默认重绘幅度（%），`-d` 可覆盖 |
| `ref_max_side` | `1536` | 参考图最长边，超过等比缩小 |
| `default_preset` | 空 | 默认预设；**留空**则说到触发词才套 |
| `workflow_rules` | 空 | 关键词路由，每行 `工作流名: 关键词1, 关键词2` |
| `prompt_template` | `{base}, {tags}, {prompt}` | 提示词拼接模板 |
| `extra_workflow_dirs` | 空 | 额外工作流目录，`;` 分隔 |
| `max_concurrency` | `1` | 并发出图上限，显存紧张就保持 1 |
| `width` / `height` | `0` | 覆盖尺寸，0 = 沿用工作流 |
| `seed` | `-1` | `-1` 每次随机 |
| `timeout` | `300` | 单次生成超时（秒） |
| `check_output_writable` | `true` | 提交前探测输出目录可写，省得白等 |
| `cache_limit` | `200` | 本地图片缓存张数上限 |
| `verbose_replies` | `false` | 打开后回执说详细内容（细节本来就在日志里） |
| `persona_enable` / `persona_file` | `true` / 空 | 人设台词开关与文件路径 |
| `nsfw_block_prompt` | `true` | 生成前拦截露骨提示词 |
| `nsfw_filter` | `true` | 生成后检查产出图（启发式兜底） |
| `nsfw_skin_threshold` | `55` | 暴露度阈值，实际除以 100；越低越严 |
| `nsfw_allow_private_r18` | `true` | 私聊 + R18 白名单才放行露骨产出 |
| `blocked_limit` | `20` | `blocked/` 最多留几张（0 = 不淘汰） |
| `rate_limit_enable` | `true` | 频率限制总开关 |
| `rate_limit_count` / `rate_limit_window` / `rate_limit_cooldown` | `5` / `600` / `10` | 见上 |
| `draw_log_max_kb` | `512` | 出图日志单文件上限，超过轮转 |
| `enable_r18` / `r18_user_whitelist` | `false` / 空 | 露骨内容**豁免权限**门（不是素材开关）：私聊且名单内才不被内容门拦。本发行版无任何露骨素材 |
| `enable_group` / `enable_private` | `true` | 群/私聊开关 |
| `group_whitelist` / `user_blacklist` | 空 | 名单限制 |

> **配置升级提示**：AstrBot 的插件配置是**首次加载时按当时的默认值生成**的，
> 之后升级插件**不会**自动补齐新键。所以升级后如果发现某个新配置项在 WebUI 里
> 看不到，把 `<AstrBot>/data/config/astrbot_plugin_comfyui_local_config.json`
> 里的新键手动补上（或删掉这个文件让它重新生成，注意会丢你改过的值）。
> 代码里对缺失的键都有兜底，不会因此报错。

---

## 七、故障排查

| 现象 | 处理 |
|---|---|
| 「连不上 ComfyUI」 | 确认 ComfyUI 已启动；`/画图 状态` 看具体错误 |
| 「引用的模型本机没有」 | 工作流里的模型名要和你本机一致 —— 别人分享的工作流要先改模型名 |
| 「未知的 class_type」 | 该工作流依赖的自定义节点没装，先在 ComfyUI 里装好 |
| 出图跑到最后一步报写盘失败 | ComfyUI 的输出目录没写权限，或磁盘满了 |
| 角色不像 / 提示词没生效 | `/画图 预设信息 预设名` 看基础提示词有没有解析出来 |
| 图被拦下了 | 内容兜底的启发式会误拦（见第五章）。图在 `blocked/` 里，可调高 `nsfw_skin_threshold` |
| 提示「N 分钟内已经画了 M 张」 | 频率限制，改 `rate_limit_count` / `rate_limit_window` 或等窗口滑动 |
| 中文说了没用 | 底层只认英文标签，看聊天里提示的「没翻成英文」的词 |

---

## 八、已知限制

* **产出图的内容检查是启发式**，不是分类器：能抓明显事故，但会误拦也会漏放。
* **参考图会锁住姿势和背景**，想去掉就靠插件自动判断或自己别写 `-r`；去掉后
  配色也会漂。
* **中文只有常用说法有内置翻译**，生僻说法等于没写。
* **img2img 无法 100% 复刻参考图细节**（底模倾向，改不掉）。
* **视频类产出只回报文件名**，不投递视频本身，也**没有内容检查** —— 如果你要用
  视频工作流，请自行确保内容合规。
* **内容门只认「明确的露骨描述」**，暗示、黑话、绕开说法一律挡不住；
  `enable_r18` + `r18_user_whitelist` 是给白名单用户私聊**免检**的权限门，
  它只决定「拦不拦」，不会让插件产出本就没有的东西。

---

## 九、目录结构

```
astrbot_plugin_comfyui_local/
├── main.py              插件主体（指令、LLM 工具、工作流/预设/参考图管理、生成调度）
├── uigraph.py           界面格式 → API 格式转换器
├── comfy_client.py      HTTP + WebSocket 客户端
├── presets.py           预设系统
├── references.py        参考图处理
├── safety.py            内容兜底（像素判据 + 关键词表）
├── guard.py             频率限制 + 出图日志
├── persona.py           人设台词
├── _conf_schema.json    WebUI 配置项
├── metadata.yaml        插件元信息
├── requirements.txt     依赖（aiohttp）
└── workflows/           首次运行的播种模板（空的，照注释填）
```

运行时数据（**不在**插件目录里）：

```
<AstrBot>/data/plugin_data/astrbot_plugin_comfyui_local/
├── workflows/    实际生效的工作流（你导出的 json 放这里）
├── presets.json  实际生效的预设
├── refs/         参考图
├── persona.json  回执台词
├── cache/        出图缓存（cache_limit 自动淘汰）
├── blocked/      被内容兜底拦下的图（blocked_limit 自动淘汰）
└── draw.log      出图日志
```

---

## 许可

MIT License，见 [LICENSE](LICENSE)。Copyright (c) 2026 Fangnai-byte。

简要说：你可以随便用、改、再分发（包括商用），只要保留版权与许可声明；
软件按原样提供，出图内容由使用者自己负责。
