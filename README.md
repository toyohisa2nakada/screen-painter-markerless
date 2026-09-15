# Screen Painter – markerless (案B: スマホは最小、PC ブラウザで全部)

スマホのカメラ映像を WebRTC (PeerJS) で PC に送り、PC のブラウザが
「いま画面に出している内容」と照合してホモグラフィ（スマホ画像座標 → 画面座標）を推定し、
スマホでなぞった位置に画面へ描画します。四隅マーカーは不要で、画面の一部しか映っていなくても、
画面が撮影画像の一部でも動きます。

```
スマホ (phone.html)                      PC (pc.html)
  getUserMedia ─── WebRTC video ───▶  <video> ─▶ XFeat(ONNX, WebGPU) ─┐
  touch + clock ── DataChannel ────▶  onData                          ├─▶ mutual-NN ─▶ RANSAC H ─▶ 平滑化
                                       game canvas ─▶ XFeat(pyramid) ┘         ▲
                 PeerJS signaling                                              └─ 参照リングバッファ (時刻付き)
  ◀────────── Node: server.js (静的配信 + PeerServer + HTTPS) ──────────▶
```

推論はすべて PC 側ブラウザ内（onnxruntime-web / WebGPU、無ければ WASM）。Python は不要です。

## セットアップ

1. Node.js 18 以上。プロジェクトフォルダで:

   ```
   npm install
   ```

2. HTTPS 証明書（スマホのカメラは HTTPS 必須）。mkcert を使う場合:

   ```
   # 一度だけ: ローカル CA を作って PC に登録
   mkcert -install
   # PC の LAN IP (例 192.168.1.20) と localhost 用の証明書を certs/ に作る
   mkcert -cert-file certs/cert.pem -key-file certs/key.pem 192.168.1.20 localhost
   ```

   スマホ側には CA を入れる必要があります。`mkcert -CAROOT` で表示されるフォルダの `rootCA.pem` を
   スマホに送って（AirDrop / メール等）インストールし、iOS は
   設定 → 一般 → 情報 → 証明書信頼設定 で「完全な信頼」を ON、Android は
   設定 → セキュリティ → 証明書のインストール → CA 証明書 から入れます。
   LAN の IP が変わったら証明書を作り直してください。

   証明書が無い場合はサーバが HTTP で起動します（PC 側の動作確認やテストには使えますが、
   スマホのカメラは開けません）。

3. 起動:

   ```
   npm start
   ```

   起動ログに `PC page : https://192.168.x.x:8443/pc.html` のように URL が出ます。

## 使い方

1. PC の Chrome / Edge で `https://<PCのIP>:8443/pc.html` を開く（`localhost` でも動きますが、
   QR に載る URL が LAN IP になるよう IP で開くのがおすすめ）。右上のパネルに「モデル準備完了」と出ます。
2. スマホで右上の QR コードを読み `phone.html` を開き、「カメラを開始」を押す。
3. スマホで PC 画面を映すと、パネルの統計が `TRACKING` になり、デバッグ表示に
   検出した画面の枠（シアン）と inlier（緑点）が出ます。
4. スマホの画面上をなぞると PC 画面に線が描かれます。

パネルの項目:

- 背景テクスチャ: デモ画面の背景模様の ON/OFF。平坦な UI で精度がどう落ちるかを試せます。
- 映像遅延補正: 受信フレームの撮影時刻を `受信時刻 − RTT/2 − この値` と見積もり、その時刻に
  近い参照フレームを選びます（画面が動くゲームで効きます）。
- 最小 inlier: これ未満の推定は捨てます。誤検出が出るなら上げ、追跡が途切れるなら下げます。
- 平滑化: 推定した四隅の指数移動平均。0 で生の推定。
- `D` キーでパネルを隠せます。

## 仕組み（コードの対応）

| ファイル | 役割 |
|---|---|
| `server.js` | express 静的配信、`/peerjs` に PeerServer、`certs/` があれば HTTPS、QR 生成 |
| `public/pc.html`, `public/js/pc.js` | デモ画面の描画、PeerJS 受信、参照特徴のリングバッファ、H 推定・検証・平滑化、タッチ描画、デバッグ表示 |
| `public/phone.html`, `public/js/phone.js` | カメラ取得（`contentHint='detail'`、maxBitrate 2.5Mbps、maintain-resolution）、`peer.call`、タッチとクロック同期の DataChannel、再接続 |
| `public/js/feature-worker.js` | Worker で onnxruntime-web を動かす。XFeat backbone の推論と、記述子の類似度行列 (GPU MatMul) |
| `public/js/xfeat.js` | XFeat の後処理を JS で再実装: NMS、信頼度スコア、top-k、サブピクセル、bicubic 記述子サンプリング、相互最近傍 |
| `public/js/homography.js` | 正規化 DLT + RANSAC + inlier 再推定（依存ライブラリなし） |
| `public/js/matcher.js` | 上記をつなぐ補助（画像→グレースケール、ピラミッド結合、推定、四隅の妥当性検査） |
| `public/js/demo-game.js` | 動くデモ画面 |
| `models/xfeat_web.onnx` | XFeat backbone (2.6MB)。`tools/export_onnx_web.py` で公式 PyTorch 重みから export |
| `models/mnn_match.onnx` | 記述子の内積行列を計算するだけの小さな ONNX（WebGPU で回すため） |

ポイント:

- **参照は固定ではなく毎フレームの画面そのもの。** 120ms ごとに画面 canvas を取り込み、
  変化があれば特徴を抽出して時刻付きで保持（直近 16 枚）。変化が無ければ最後の参照の有効期間を延ばすだけ。
- **スケールピラミッド。** XFeat は単一スケールなので、画面側は 1280 / 640 / 320 px 幅の 3 段で特徴を
  出して連結しています。これで「近づいて一部だけ映す」(拡大) と「離れて画面が小さく映る」(縮小) の両方に
  対応します（ピラミッド無しだと拡大側が破綻します）。
- **時刻合わせ。** DataChannel の ping/pong で RTT とクロック差を推定し、フレームの推定撮影時刻に近い
  参照から順に最大 3 枚試して inlier が最多のものを採用。
- **検証と平滑化。** inlier 数の下限、投影した四隅が凸で向きが保たれているか、を通ったものだけ採用し、
  四隅を EMA で平滑化してから H を作り直します。タッチはタッチ時刻に最も近い H で変換します。

## テスト

`test/` にヘッドレス Chromium (Playwright) のテストがあります（Python 3 + `pip install playwright`、
サーバを `PORT=8080 npm start` で起動しておく）。

- `python test/run_synth.py http://localhost:8080/test/synth.html?gpu=0`
  既知のホモグラフィで画面を歪めた合成画像から H を復元し、誤差を測ります（近接 / 遠方 / 遠方+回転）。
- `python test/run_e2e.py http://localhost:8080`
  PC ページとスマホページを同じブラウザで開き、偽カメラ（歪めた画面画像）で本物の PeerJS/WebRTC 接続を
  通し、タッチが画面中央に描かれることを確認します。

参考値（クラウドの 2 コア CPU、WASM）: 抽出 140ms、マッチ+RANSAC 70ms。WebGPU が使える PC では
一桁 ms 台になります。合成テストの誤差は近接で 0.4px、遠方（画面が 370px 幅で映る）で 3〜10px（画面座標）。

## 既知の制限・次の一手

- 平坦な UI（テクスチャ OFF）では inlier が大きく減ります。実アプリでは背景に薄い模様を合成するか、
  描いたストローク自体が特徴になることを利用してください。
- 参照抽出（3 段）は WebGPU 前提の重さです。WASM しか無い環境では `refLevels` を `[640, 320]` に、
  `refIntervalMs` を大きくしてください（`pc.js` 冒頭の `S`）。
- 精度を上げるなら: LightGlue（学習済みマッチャー）への差し替え、前フレームの H で参照をワープして
  同一スケールで再照合する追跡モード、画面の外枠（ベゼル）検出の併用。
- 現状は Chrome / Edge を想定。Safari では WebGPU の可否と `requestVideoFrameCallback` の挙動を要確認。

## ONNX の再 export

```
git clone https://github.com/verlab/accelerated_features
cp tools/export_onnx_web.py accelerated_features/
cd accelerated_features && pip install torch onnx && python export_onnx_web.py ../models/xfeat_web.onnx
```
