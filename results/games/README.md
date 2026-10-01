# Game results

Everything behind the game numbers in [docs/results.md](../../docs/results.md). Run names are replaced by the
released model names, and the harnesses' `jev_*` field names, which mean "the model under test", are renamed `model_*`.

| Path | Contents | Harness |
|---|---|---|
| `chess/elo_ladder/<model>/` | every Elo ladder game as JSON, all games as `all_games.pgn`, and the fitted rating | wondertwins/jev-benchmark |
| `chess/elo_ladder_rich/<player>/` | the rich-level Elo ladder games of StartLux-Decision-27B and Jev 1.13, all games as `all_games.pgn`, and the fitted ratings | wondertwins/jev-benchmark |
| `chess/match/<condition>/` | the 1,024 games of StartLux-Decision-27B against Jev 1.13 in four conditions, as `games.pgn` and `games.json`; `chess/match/summary.json` has the scores, openings and Chess960 positions | wondertwins/jev-benchmark, through [demos/chess](../../demos/chess) |
| `chess/positions/<model>/` | move choice at four state levels (A), mate in one (E) and the harness's other probes (B, C, D, F) | wondertwins/jev-benchmark |
| `npc_addressee/<model>.json` | every utterance in the clean, speech-to-text and misheard variants, with a probability per character | wondertwins/jev-benchmark |
| `arcade/dino.json` | obstacles cleared in every Dino Run | surafel-kindu/system-one-models-game-test |
