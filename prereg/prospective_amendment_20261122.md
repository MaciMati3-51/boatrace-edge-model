# 前向き検証の変更（2026-11-22）— BoatraceCSV の公開停止への対応

対象: v4 の主判定（`prereg/live_v4_design.md` §4）と v5b の副判定（`prereg/live_v5b_design.md` §4）。期間はどちらも
**20261113〜20270112 のまま**。モデル・係数・判定の式は変えない（`prereg/live_v4_freeze.json`・`prereg/live_v5_freeze.json` の
ハッシュを照合する）。

**この文書の保存時点で、20261113〜20261121 の公式 B/K・boatraceopenapi・turnmark はローカルにあるが、v3・v4・v5b の予測も
評価も一度も行っていない。** 取得は全て v5b の凍結（`prereg/live_v5_freeze.json`）の後。

## 1. 何が起きたか

BoatraceCSV（boatracecsv.github.io）は 2026-11-15 にデータ公開を停止して非公開化した（GitHub `BoatraceCSV/fun-site` PR #18）。
20261113 以降の tkz・stt・sui・od2 は全て404で、過去のファイルも消えた。事前登録の `evaluate_prospective_v4.py`・
`evaluate_prospective_v5.py` は od2・tkz・stt を前提にしているので、そのままでは回せない。

## 2. 代わりのデータ

| 失ったもの | 代わり | 照合 |
|---|---|---|
| tkz の展示タイム（標本の条件） | boatraceopenapi/api の `preview.racers.*.exhibition_time` | 2026-10-01〜10 の 1,536R で tkz と 9,083艇全て一致（tkz 側の欠測120艇は openapi にある） |
| stt の展示進入・展示ST（v5b の入力） | 同 `course_number`・`start_timing` | 同期間で 9,204艇とも全て一致。レース単位の進入も全一致（変更あり 12.8%） |
| od2（締切約9分前の板） | **無し** | — |
| （新規）確定オッズ全組 | turnmark/api の `odds`（全7券種） | 2026-11-21 で的中目の 払戻/100 と 501/504 一致 |

どちらも非公式・MIT。boatraceopenapi の過去日の preview は取り直した後の最終値で、撮影時刻が無い。展示の値はスタート展示の後に
変わらないので、上の照合（全艇一致）をもって代用とする。

## 3. 変更点（`src/evaluate_prospective_amended.py`）

1. **標本**: 「締切前60分以内に撮った展示タイムが K と6艇とも一致」→「boatraceopenapi の展示タイムが K と6艇とも一致」
2. **v5b の展示の進入・ST**: stt → boatraceopenapi の preview
3. **測れなくなるもの**: Q3（締切前の板との合成）と Q4（紙上回収率）。結果ファイルに「測れない」と記録する。回収率の前向き判定は
   この期間では行わない
4. **副指標の追加**: turnmark の確定オッズ全組から、正規化子を仮置きしない確定市場の log-loss（3連複の 1.345 仮置きが要らなくなる）。
   オッズ 0 の組（票なし）は正規化に入れず、的中目が 0 のレースは除く

判定の式はそのまま:
- v4: Q1 = v4 − v3 の 2連単・2連複の 95%上限 < 0 なら「v4 が良い」、Q2 = v4 − 確定市場（払戻由来）の各券種の 95%上限 < 0
- v5b: v5b − v4 の 2連単・3連単の 95%上限 < 0 なら「v5b が良い」

配線は 2026-10-01〜10（既知の期間、v4a・v3 の凍結モデル）で試運転した（結果は判定に使わない）。1,419R、展示タイムの不一致0艇、
確定市場の log-loss は払戻由来と全組由来で 3連単 −0.0043 / −0.0048 と近い。

## 4. 取得の運用

`src/acquire_openapi.py --start YYYY-MM-DD --end YYYY-MM-DD`（両ソース、1件/秒、終わった日だけ）。公式 B/K は従来どおり
`download_lzh.py` → `extract_lzh.py` → `import_bk_range.py` → `parse_b_meta.py` → `v4_meta.py`。評価の前に `v3_features.py` で
特徴量のキャッシュを 20270112 まで延ばす。
