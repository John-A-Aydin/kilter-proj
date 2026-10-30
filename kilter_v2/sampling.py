"""Generation from nothing or from given holds. web/src/decoding.mjs mirrors ``allowed_tokens``."""
import numpy as np
import torch

from .data import BOS, EOS, FINISH, FOOT, MAX_FINISHES, MAX_STARTS, PAD, SEP, START, Board


def validate_given(board, given, max_holds):
    seen, tokens = set(), []
    for hold, role in given:
        if hold not in board.dense or role not in range(4) or hold in seen:
            raise ValueError("Given holds must be unique board holds with roles 0-3")
        seen.add(hold)
        tokens.append(board.token(board.dense[hold], role))
    roles = [role for _, role in given]
    if roles.count(START) > MAX_STARTS or roles.count(FINISH) > MAX_FINISHES:
        raise ValueError("A climb has at most two start holds and two finish holds")
    if len(given) + (START not in roles) + (FINISH not in roles) > max_holds:
        raise ValueError(f"Too many given holds for a {max_holds}-hold climb")
    return sorted(tokens, key=lambda t: board.rank[t])


def allowed_tokens(board, used, last_rank, starts, finishes, count, max_holds):
    """Continuations every training climb satisfies: canonical order, unique holds, 1-2 starts/finishes."""
    allowed = np.zeros(board.vocab_size, dtype=bool)
    holds = np.arange(board.vocab_size - 4) // 4
    roles = np.arange(board.vocab_size - 4) % 4
    candidate = (board.rank[4:] > last_rank) & ~np.isin(holds, list(used))
    candidate &= (roles != START) | (starts < MAX_STARTS)
    candidate &= (roles != FINISH) | (finishes < MAX_FINISHES)
    needed = (starts == 0) + (finishes == 0)
    if starts == 0:
        candidate &= roles == START
    elif finishes == 0:
        # Feet come after finishes in canonical order, so a finish must come first.
        candidate &= roles != FOOT
        if max_holds - count <= needed:
            candidate &= roles == FINISH
    if count >= max_holds:
        candidate[:] = False
    allowed[4:] = candidate
    allowed[EOS] = needed == 0
    return allowed


@torch.no_grad()
def generate(model, board, grade, angle, given=(), temperature=1.0, top_k=0, generator=None,
             constrained=True, max_holds=40):
    """Sample the remaining holds. ``given`` is [(hold_id, role)]; returns holds, given first."""
    if grade < 0 or not 0 <= angle <= 90 or temperature <= 0 or top_k < 0:
        raise ValueError("Invalid grade, angle, temperature, or top_k")
    max_holds = min(max_holds, model.max_len - 2)
    given_tokens = validate_given(board, given, max_holds)
    device = next(model.parameters()).device
    conditions = torch.tensor([[grade, angle / 90.0]], device=device)
    tokens = [BOS, *given_tokens, SEP]
    used = {board.split(t)[0] for t in given_tokens}
    roles = [board.split(t)[1] for t in given_tokens]
    starts, finishes, last_rank, generated = roles.count(START), roles.count(FINISH), -1, []
    ended = False
    while len(given_tokens) + len(generated) <= max_holds:
        logits = model(torch.tensor([tokens], device=device), conditions)[0, -1].float() / temperature
        if constrained:
            allowed = allowed_tokens(board, used, last_rank, starts, finishes,
                                     len(given_tokens) + len(generated), max_holds)
            logits[~torch.from_numpy(allowed).to(device)] = -float("inf")
        else:
            logits[[PAD, BOS, SEP]] = -float("inf")
            if len(given_tokens) + len(generated) >= max_holds:
                logits[4:] = -float("inf")
        if top_k:
            threshold = logits.topk(min(top_k, logits.numel())).values[-1]
            logits[logits < threshold] = -float("inf")
        token = torch.multinomial(logits.softmax(-1), 1, generator=generator).item()
        tokens.append(token)
        if token == EOS:
            ended = True
            break
        dense, role = board.split(token)
        used.add(dense)
        starts += role == START
        finishes += role == FINISH
        last_rank = board.rank[token]
        generated.append(token)
    holds = ([{**h, "given": True} for h in board.decode(given_tokens)]
             + [{**h, "given": False} for h in board.decode(generated)])
    return {"v_grade": grade, "angle": angle, "normalized_angle": angle / 90.0,
            "ended_with_eos": ended, "holds": holds}


def climb_checks(holds, y_of):
    """Structural and physical-plausibility summary of one climb (list of hold dicts)."""
    ids = [h["hold_token"] for h in holds]
    by_role = {r: sorted(y_of[h["hold_token"]] for h in holds if h["hold_use_type"] == r) for r in range(4)}
    route = sorted(y_of[h["hold_token"]] for h in holds if h["hold_use_type"] != FOOT)
    checks = {
        "no_repeated_hold": len(set(ids)) == len(ids),
        "starts_1_2": 1 <= len(by_role[START]) <= MAX_STARTS,
        "finishes_1_2": 1 <= len(by_role[FINISH]) <= MAX_FINISHES,
        "start_below_finish": bool(by_role[START] and by_role[FINISH]) and by_role[START][-1] < by_role[FINISH][0],
        "has_feet": bool(by_role[FOOT]),
    }
    checks["valid"] = all(checks.values())
    return checks, {"holds": len(holds), "feet": len(by_role[FOOT]),
                    "finish_height": by_role[FINISH][-1] if by_role[FINISH] else float("nan"),
                    "largest_hand_gap": float(np.diff(route).max()) if len(route) > 1 else float("nan")}


__all__ = ["Board", "allowed_tokens", "climb_checks", "generate", "validate_given"]
