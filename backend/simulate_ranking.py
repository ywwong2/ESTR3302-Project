from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def softmax(values: list[float]) -> list[float]:
    if not values:
        return []
    m = max(values)
    exps = [math.exp(v - m) for v in values]
    total = sum(exps)
    if total <= 0:
        return [1.0 / len(values)] * len(values)
    return [v / total for v in exps]


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def l2_norm(a: list[float]) -> float:
    return math.sqrt(sum(x * x for x in a))


def cosine(a: list[float], b: list[float]) -> float:
    na = l2_norm(a)
    nb = l2_norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return dot(a, b) / (na * nb)


def project_to_simplex(v: list[float]) -> list[float]:
    """Project v onto the unit simplex {w | w_i >= 0, sum w_i = 1}."""
    n = len(v)
    u = sorted(v, reverse=True)
    cssv = [0.0] * n
    running = 0.0
    rho = -1
    for i, ui in enumerate(u):
        running += ui
        cssv[i] = running
        t = (running - 1.0) / (i + 1)
        if ui - t > 0:
            rho = i
    if rho == -1:
        return [1.0 / n] * n
    theta = (cssv[rho] - 1.0) / (rho + 1)
    w = [max(x - theta, 0.0) for x in v]
    s = sum(w)
    if s == 0:
        return [1.0 / n] * n
    return [x / s for x in w]


@dataclass
class ImageState:
    image_id: int
    category: int
    embedding: list[float]
    quality: float
    freshness: float
    cosine_by_query: dict[str, float] | None = None
    tilde_e: float = 0.0
    tilde_v: float = 0.0
    tilde_l: float = 0.0
    ctr: float = 0.0
    lr: float = 0.0
    awt: float = 0.0
    social_proof: float = 0.0

    def update_stats(
        self,
        e_t: float,
        v_t: float,
        l_t: float,
        mean_watch_t: float,
        rho: float,
        c_ctr: float,
        m_ctr: float,
        c_lr: float,
        m_lr: float,
        omega_l: float,
        m_social: float,
    ) -> None:
        self.tilde_e = (1.0 - rho) * self.tilde_e + rho * e_t
        self.tilde_v = (1.0 - rho) * self.tilde_v + rho * v_t
        self.tilde_l = (1.0 - rho) * self.tilde_l + rho * l_t

        self.ctr = (self.tilde_v + c_ctr * m_ctr) / (self.tilde_e + c_ctr)
        self.lr = (self.tilde_l + c_lr * m_lr) / (self.tilde_v + c_lr)
        self.awt = (1.0 - rho) * self.awt + rho * mean_watch_t

        numerator = math.log1p(self.tilde_v + omega_l * self.tilde_l)
        self.social_proof = min(1.0, numerator / m_social) if m_social > 0 else 0.0


@dataclass
class UserState:
    user_id: int
    xi_view: float
    xi_watch: float
    xi_like: float


def build_category_prototypes(num_categories: int, dim: int, rng: random.Random) -> list[list[float]]:
    prototypes: list[list[float]] = []
    for _ in range(num_categories):
        vec = [rng.gauss(0, 1) for _ in range(dim)]
        norm = l2_norm(vec)
        prototypes.append([x / norm for x in vec])
    return prototypes


def noisy_query(base: list[float], rng: random.Random, noise_std: float) -> list[float]:
    vec = [base[i] + rng.gauss(0, noise_std) for i in range(len(base))]
    norm = l2_norm(vec)
    if norm == 0:
        return base[:]
    return [x / norm for x in vec]


def setup_images(
    num_images: int,
    num_categories: int,
    embedding_dim: int,
    rng: random.Random,
) -> tuple[list[ImageState], list[list[float]]]:
    prototypes = build_category_prototypes(num_categories, embedding_dim, rng)

    images: list[ImageState] = []
    for idx in range(num_images):
        category = idx % num_categories
        base = prototypes[category]
        emb = noisy_query(base, rng, noise_std=0.08)
        quality = rng.betavariate(2.0, 5.0)
        freshness = rng.uniform(0.2, 1.0)
        images.append(
            ImageState(
                image_id=idx,
                category=category,
                embedding=emb,
                quality=quality,
                freshness=freshness,
                ctr=0.1,
                lr=0.05,
            )
        )
    return images, prototypes


def setup_users(num_users: int, rng: random.Random) -> list[UserState]:
    users: list[UserState] = []
    for uid in range(num_users):
        users.append(
            UserState(
                user_id=uid,
                xi_view=rng.gauss(0, 0.1),
                xi_watch=rng.gauss(0, 0.1),
                xi_like=rng.gauss(0, 0.1),
            )
        )
    return users


def setup_images_from_cache(
    cache_path: Path,
    query_labels: list[str],
    rng: random.Random,
) -> list[ImageState]:
    data = json.loads(cache_path.read_text(encoding="utf-8"))
    images_raw = data.get("images", [])
    if not images_raw:
        raise ValueError(f"No images found in cache file: {cache_path}")

    label_to_idx = {label: i for i, label in enumerate(query_labels)}
    images: list[ImageState] = []

    for idx, raw in enumerate(images_raw):
        category_label = str(raw.get("category", "")).lower().strip()
        if category_label not in label_to_idx:
            raise ValueError(
                f"Unknown category '{category_label}' in cache. "
                f"Expected one of {query_labels}."
            )

        cosine_by_query = {
            str(k): float(v)
            for k, v in dict(raw.get("cosine_by_query", {})).items()
            if str(k) in label_to_idx
        }
        missing = [q for q in query_labels if q not in cosine_by_query]
        if missing:
            raise ValueError(
                f"Image entry at index {idx} is missing cosine values for queries: {missing}"
            )

        images.append(
            ImageState(
                image_id=idx,
                category=label_to_idx[category_label],
                embedding=[],
                quality=rng.betavariate(2.0, 5.0),
                freshness=rng.uniform(0.2, 1.0),
                ctr=0.1,
                lr=0.05,
                cosine_by_query=cosine_by_query,
            )
        )

    return images


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run image-ranking behavior simulation with policy gradient.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-images", type=int, default=50)
    parser.add_argument("--num-categories", type=int, default=5)
    parser.add_argument("--embedding-dim", type=int, default=32)

    parser.add_argument("--num-users", type=int, default=1000)
    parser.add_argument("--rounds", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=10)

    parser.add_argument("--rho", type=float, default=0.1)
    parser.add_argument("--m-ctr", type=float, default=0.1)
    parser.add_argument("--m-lr", type=float, default=0.05)
    parser.add_argument("--c-ctr", type=float, default=10.0)
    parser.add_argument("--c-lr", type=float, default=10.0)
    parser.add_argument("--omega-l", type=float, default=5.0)
    parser.add_argument("--m-social", type=float, default=math.log1p(500.0))

    parser.add_argument("--gamma-pos", type=float, default=1.0)
    parser.add_argument("--kappa", type=float, default=10.0)

    parser.add_argument("--eta", type=float, default=0.01)
    parser.add_argument("--tau", type=float, default=5.0)
    parser.add_argument("--beta", type=float, default=0.0)

    parser.add_argument("--lambda-v", type=float, default=1.0)
    parser.add_argument("--lambda-t", type=float, default=2.0)
    parser.add_argument("--lambda-l", type=float, default=5.0)

    parser.add_argument("--out-dir", type=str, default="backend/data/simulation")
    parser.add_argument(
        "--dataset-cache",
        type=str,
        default="",
        help="Path to dataset cosine cache JSON. If set, simulation uses cache-derived image-query cosines.",
    )
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--diagnostics", action="store_true", help="Write extra diagnostics CSV with feature stats, gradient norms, and pre/post projection weights.")
    parser.add_argument("--normalize-features", action="store_true", help="Min-max normalize ranking features to [0,1] before scoring.")
    parser.add_argument("--drop-social", action="store_true", help="Remove social_proof from the ranking feature vector.")
    parser.add_argument("--freeze-social", type=float, default=-1.0, help="If >=0, fix social weight to this value and optimize remaining dims on simplex.")
    parser.add_argument("--freeze-cos", type=float, default=-1.0, help="If >=0, fix cosine weight to this value and optimize remaining dims on simplex.")
    parser.add_argument("--lagged-social", action="store_true", help="Use previous-round social proof instead of current-round.")
    parser.add_argument("--item-diagnostics", action="store_true", help="Write per-image stats and rank trajectories to item_diagnostics.csv.")
    return parser


def run_simulation(args: argparse.Namespace | None = None, **kwargs) -> dict[str, Path]:
    if args is None:
        parser = get_parser()
        args = parser.parse_args([])
    # Override with any kwargs passed directly (useful for ablation script)
    for k, v in kwargs.items():
        if hasattr(args, k):
            setattr(args, k, v)
        else:
            # Support kebab-case to snake_case mapping
            k_snake = k.replace("-", "_")
            if hasattr(args, k_snake):
                setattr(args, k_snake, v)
    rng = random.Random(args.seed)
    query_labels = ["cat", "dog", "plane", "clothes", "car"]

    use_dataset_cache = bool(args.dataset_cache)
    if use_dataset_cache:
        cache_path = Path(args.dataset_cache)
        images = setup_images_from_cache(cache_path, query_labels, rng)
        prototypes: list[list[float]] = []
        print(f"Loaded dataset cache: {cache_path} ({len(images)} images)")
    else:
        images, prototypes = setup_images(
            num_images=args.num_images,
            num_categories=args.num_categories,
            embedding_dim=args.embedding_dim,
            rng=rng,
        )

    users = setup_users(args.num_users, rng)

    # h = [1, match, cos, freshness, social_proof]
    alpha = [-0.8, 0.6, 2.0, 0.3, 1.0]
    gamma = [-0.6, 0.4, 1.0, 0.2, 0.4]
    delta = [-2.0, 0.3, 0.8, 0.1, 0.5]
    alpha_q = 0.5
    gamma_q = 2.5
    delta_t = 2.0
    delta_q = 2.0

    # Determine feature dimension
    use_social = not args.drop_social
    freeze_social = args.freeze_social >= 0.0
    freeze_cos = args.freeze_cos >= 0.0
    dim = 5 if (not use_social) else 6

    if not use_social:
        w = [1.0 / 5.0] * 5
    else:
        w = [1.0 / 6.0] * 6
        if freeze_social:
            frozen = args.freeze_social
            w[5] = frozen
            w[:5] = [x / sum(w[:5]) * (1.0 - frozen) for x in w[:5]]
        if freeze_cos:
            frozen = args.freeze_cos
            w[0] = frozen
            rem = w[1:]
            rem = [x / sum(rem) * (1.0 - frozen) for x in rem]
            w[1:] = rem

    # Feature labels for diagnostics / plotting
    feat_labels = ["cos", "fresh", "lr", "ctr", "awt"]
    if use_social:
        feat_labels.append("social")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "simulation_rounds.csv"

    headers = [
        "round",
        "avg_reward",
        "avg_exam",
        "avg_view",
        "avg_like",
        "global_ctr",
        "global_lr",
        "global_awt",
        "w0_cos",
        "w1_fresh",
        "w2_lr",
        "w3_ctr",
        "w4_awt",
        "w5_social",
    ]

    diag_headers = [
        "round", "avg_reward",
        "feat_mean_cos", "feat_std_cos",
        "feat_mean_fresh", "feat_std_fresh",
        "feat_mean_lr", "feat_std_lr",
        "feat_mean_ctr", "feat_std_ctr",
        "feat_mean_awt", "feat_std_awt",
        "feat_mean_social", "feat_std_social",
        "grad_norm", "grad_cos", "grad_fresh", "grad_lr", "grad_ctr", "grad_awt", "grad_social",
        "wpre_cos", "wpre_fresh", "wpre_lr", "wpre_ctr", "wpre_awt", "wpre_social",
        "wpost_cos", "wpost_fresh", "wpost_lr", "wpost_ctr", "wpost_awt", "wpost_social",
    ]

    diag_path = out_dir / "diagnostics.csv" if args.diagnostics else None
    diag_writer = None
    if diag_path:
        diag_f = diag_path.open("w", newline="", encoding="utf-8")
        diag_writer = csv.DictWriter(diag_f, fieldnames=diag_headers)
        diag_writer.writeheader()

    rewards_history: list[float] = []
    weights_history: list[list[float]] = []

    # For lagged social, store previous round values
    prev_social = [0.0 for _ in images]

    item_diag_path = out_dir / "item_stats.csv" if args.item_diagnostics else None
    item_diag_f = None
    item_diag_writer = None
    if item_diag_path:
        item_diag_f = item_diag_path.open("w", newline="", encoding="utf-8")
        item_diag_writer = csv.writer(item_diag_f)
        item_diag_writer.writerow(["round", "image_id", "category", "ctr", "lr", "awt", "social_proof", "own_query_score"])

    query_rank_path = out_dir / "query_ranks.csv" if args.item_diagnostics else None
    query_rank_f = None
    query_rank_writer = None
    if query_rank_path:
        query_rank_f = query_rank_path.open("w", newline="", encoding="utf-8")
        query_rank_writer = csv.writer(query_rank_f)
        query_rank_writer.writerow(["round", "query_label", "image_id", "category", "rank", "score"])

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()

        for t in range(1, args.rounds + 1):
            if args.lagged_social:
                prev_social = [img.social_proof for img in images]

            image_aggr = {
                img.image_id: {
                    "e": 0.0,
                    "v": 0.0,
                    "l": 0.0,
                    "watch_sum": 0.0,
                    "watch_cnt": 0.0,
                }
                for img in images
            }

            grad = [0.0] * dim
            total_reward = 0.0
            total_exam = 0.0
            total_view = 0.0
            total_like = 0.0

            batch_users = rng.sample(users, k=min(args.batch_size, len(users)))
            num_query_categories = len(query_labels) if use_dataset_cache else args.num_categories

            for user in batch_users:
                query_cat = rng.randrange(num_query_categories)
                if use_dataset_cache:
                    query_emb = []
                    query_label = query_labels[query_cat]
                else:
                    query_emb = noisy_query(prototypes[query_cat], rng, noise_std=0.05)
                    query_label = ""

                x_vectors: list[list[float]] = []
                scores: list[float] = []
                cos_values: list[float] = []
                for img in images:
                    if use_dataset_cache:
                        cos_ui = float((img.cosine_by_query or {}).get(query_label, 0.0))
                    else:
                        cos_ui = cosine(query_emb, img.embedding)
                    cos_values.append(cos_ui)
                    x = [cos_ui, img.freshness, img.lr, img.ctr, img.awt]
                    if use_social:
                        social_val = prev_social[img.image_id] if args.lagged_social else img.social_proof
                        x.append(social_val)
                    x_vectors.append(x)
                    scores.append(dot(w, x))

                if args.normalize_features:
                    num_feats = dim
                    mins = [min(x_vectors[i][j] for i in range(len(images))) for j in range(num_feats)]
                    maxs = [max(x_vectors[i][j] for i in range(len(images))) for j in range(num_feats)]
                    for i in range(len(images)):
                        for j in range(num_feats):
                            den = maxs[j] - mins[j]
                            if den > 1e-8:
                                x_vectors[i][j] = (x_vectors[i][j] - mins[j]) / den
                            else:
                                x_vectors[i][j] = 0.0
                    scores = [dot(w, x_vectors[i]) for i in range(len(images))]

                ranked_indices = sorted(range(len(images)), key=lambda i: scores[i], reverse=True)
                displayed = ranked_indices[: args.top_k]

                # Policy over all candidates (reward of non-displayed defaults to 0).
                logits = [args.tau * s for s in scores]
                a = softmax(logits)
                rewards = [0.0] * len(images)

                for rank, idx in enumerate(displayed):
                    img = images[idx]
                    eta_r = (1.0 + rank) ** (-args.gamma_pos)
                    examined = 1.0 if rng.random() < eta_r else 0.0
                    image_aggr[img.image_id]["e"] += examined
                    total_exam += examined
                    if examined == 0.0:
                        continue

                    match = 1.0 if img.category == query_cat else 0.0
                    h = [1.0, match, cos_values[idx], img.freshness, img.social_proof]

                    p_view = sigmoid(dot(alpha, h) + alpha_q * img.quality + user.xi_view)
                    view = 1.0 if rng.random() < p_view else 0.0
                    image_aggr[img.image_id]["v"] += view
                    total_view += view

                    watch = 0.0
                    like = 0.0
                    if view == 1.0:
                        mu = sigmoid(dot(gamma, h) + gamma_q * img.quality + user.xi_watch)
                        a_beta = max(args.kappa * mu, 1e-6)
                        b_beta = max(args.kappa * (1.0 - mu), 1e-6)
                        watch = rng.betavariate(a_beta, b_beta)
                        image_aggr[img.image_id]["watch_sum"] += watch
                        image_aggr[img.image_id]["watch_cnt"] += 1.0

                        p_like = sigmoid(dot(delta, h) + delta_t * watch + delta_q * img.quality + user.xi_like)
                        like = 1.0 if rng.random() < p_like else 0.0
                        image_aggr[img.image_id]["l"] += like
                        total_like += like

                    r_ui = args.lambda_v * view + args.lambda_t * watch + args.lambda_l * like
                    rewards[idx] = r_ui
                    total_reward += r_ui

                expected_reward = sum(a[i] * rewards[i] for i in range(len(images)))

                # Gradient of -E_a[R] + beta||w||^2.
                for i in range(len(images)):
                    coeff = -args.tau * a[i] * (rewards[i] - expected_reward)
                    x = x_vectors[i]
                    for j in range(dim):
                        grad[j] += coeff * x[j]

            batch_size = max(len(batch_users), 1)
            grad = [g / batch_size + 2.0 * args.beta * wj for g, wj in zip(grad, w)]
            grad_norm = l2_norm(grad)
            w_pre = [wj - args.eta * gj for wj, gj in zip(w, grad)]

            if freeze_social and use_social:
                frozen = args.freeze_social
                w_free = project_to_simplex([w_pre[j] / (1.0 - frozen + 1e-12) for j in range(5)])
                w = [w_free[j] * (1.0 - frozen) for j in range(5)] + [frozen]
            elif freeze_cos:
                frozen = args.freeze_cos
                w_free = project_to_simplex([w_pre[j] / (1.0 - frozen + 1e-12) for j in range(1, dim)])
                w = [frozen] + [w_free[j - 1] * (1.0 - frozen) for j in range(1, dim)]
            else:
                w = project_to_simplex(w_pre)

            # Apply EWMA/Bayesian smoothing updates.
            for img in images:
                agg = image_aggr[img.image_id]
                mean_watch = (
                    agg["watch_sum"] / agg["watch_cnt"] if agg["watch_cnt"] > 0 else 0.0
                )
                img.update_stats(
                    e_t=agg["e"],
                    v_t=agg["v"],
                    l_t=agg["l"],
                    mean_watch_t=mean_watch,
                    rho=args.rho,
                    c_ctr=args.c_ctr,
                    m_ctr=args.m_ctr,
                    c_lr=args.c_lr,
                    m_lr=args.m_lr,
                    omega_l=args.omega_l,
                    m_social=args.m_social,
                )

            denom = max(len(batch_users) * args.top_k, 1)
            avg_reward = total_reward / max(len(batch_users), 1)
            avg_exam = total_exam / denom
            avg_view = total_view / denom
            avg_like = total_like / denom

            global_ctr = sum(img.ctr for img in images) / len(images)
            global_lr = sum(img.lr for img in images) / len(images)
            global_awt = sum(img.awt for img in images) / len(images)

            # Diagnostics: feature stats from the last user's x_vectors (representative)
            if diag_writer and x_vectors:
                num_feats_full = 6
                means = [0.0] * num_feats_full
                stds = [0.0] * num_feats_full
                padded = []
                for xv in x_vectors:
                    if len(xv) == 5:
                        padded.append(xv + [0.0])
                    else:
                        padded.append(xv[:])
                for j in range(num_feats_full):
                    vals = [padded[i][j] for i in range(len(padded))]
                    m = sum(vals) / len(vals)
                    means[j] = m
                    stds[j] = math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))
                diag_row = {
                    "round": t,
                    "avg_reward": f"{avg_reward:.6f}",
                    "feat_mean_cos": f"{means[0]:.6f}", "feat_std_cos": f"{stds[0]:.6f}",
                    "feat_mean_fresh": f"{means[1]:.6f}", "feat_std_fresh": f"{stds[1]:.6f}",
                    "feat_mean_lr": f"{means[2]:.6f}", "feat_std_lr": f"{stds[2]:.6f}",
                    "feat_mean_ctr": f"{means[3]:.6f}", "feat_std_ctr": f"{stds[3]:.6f}",
                    "feat_mean_awt": f"{means[4]:.6f}", "feat_std_awt": f"{stds[4]:.6f}",
                    "feat_mean_social": f"{means[5]:.6f}", "feat_std_social": f"{stds[5]:.6f}",
                    "grad_norm": f"{grad_norm:.6f}",
                    "grad_cos": f"{grad[0]:.6f}",
                    "grad_fresh": f"{grad[1]:.6f}",
                    "grad_lr": f"{grad[2]:.6f}",
                    "grad_ctr": f"{grad[3]:.6f}",
                    "grad_awt": f"{grad[4]:.6f}",
                    "grad_social": f"{grad[5]:.6f}" if len(grad) > 5 else "0.000000",
                    "wpre_cos": f"{w_pre[0]:.6f}",
                    "wpre_fresh": f"{w_pre[1]:.6f}",
                    "wpre_lr": f"{w_pre[2]:.6f}",
                    "wpre_ctr": f"{w_pre[3]:.6f}",
                    "wpre_awt": f"{w_pre[4]:.6f}",
                    "wpre_social": f"{w_pre[5]:.6f}" if len(w_pre) > 5 else "0.000000",
                    "wpost_cos": f"{w[0]:.6f}",
                    "wpost_fresh": f"{w[1]:.6f}",
                    "wpost_lr": f"{w[2]:.6f}",
                    "wpost_ctr": f"{w[3]:.6f}",
                    "wpost_awt": f"{w[4]:.6f}",
                    "wpost_social": f"{w[5]:.6f}" if len(w) > 5 else "0.000000",
                }
                diag_writer.writerow(diag_row)

            writer.writerow(
                {
                    "round": t,
                    "avg_reward": f"{avg_reward:.6f}",
                    "avg_exam": f"{avg_exam:.6f}",
                    "avg_view": f"{avg_view:.6f}",
                    "avg_like": f"{avg_like:.6f}",
                    "global_ctr": f"{global_ctr:.6f}",
                    "global_lr": f"{global_lr:.6f}",
                    "global_awt": f"{global_awt:.6f}",
                    "w0_cos": f"{w[0]:.6f}",
                    "w1_fresh": f"{w[1]:.6f}",
                    "w2_lr": f"{w[2]:.6f}",
                    "w3_ctr": f"{w[3]:.6f}",
                    "w4_awt": f"{w[4]:.6f}",
                    "w5_social": f"{w[5]:.6f}" if len(w) > 5 else "0.000000",
                }
            )

            rewards_history.append(avg_reward)
            weights_history.append(w[:])

            # Item-level diagnostics
            if args.item_diagnostics:
                # Compute scores and ranks for each query
                tracked_queries = query_labels if use_dataset_cache else [f"cat_{i}" for i in range(args.num_categories)]
                for q_label in tracked_queries:
                    scores_q: list[tuple[float, int, int]] = []
                    for img in images:
                        if use_dataset_cache:
                            cos_ui = float((img.cosine_by_query or {}).get(q_label, 0.0))
                        else:
                            cos_ui = 0.0
                        x = [cos_ui, img.freshness, img.lr, img.ctr, img.awt]
                        if use_social:
                            x.append(img.social_proof)
                        s = dot(w, x)
                        scores_q.append((s, img.image_id, img.category))
                    scores_q.sort(key=lambda x: x[0], reverse=True)
                    for rank, (s, iid, cat) in enumerate(scores_q, start=1):
                        query_rank_writer.writerow([t, q_label, iid, cat, rank, f"{s:.6f}"])

                # Write per-image stats with own-query score
                for img in images:
                    if use_dataset_cache:
                        own_label = tracked_queries[img.category] if img.category < len(tracked_queries) else tracked_queries[0]
                        cos_ui = float((img.cosine_by_query or {}).get(own_label, 0.0))
                    else:
                        cos_ui = 0.0
                    x = [cos_ui, img.freshness, img.lr, img.ctr, img.awt]
                    if use_social:
                        x.append(img.social_proof)
                    own_score = dot(w, x)
                    item_diag_writer.writerow([t, img.image_id, img.category, f"{img.ctr:.6f}", f"{img.lr:.6f}", f"{img.awt:.6f}", f"{img.social_proof:.6f}", f"{own_score:.6f}"])

            if t % 50 == 0 or t == 1 or t == args.rounds:
                print(
                    f"round={t:4d} reward={avg_reward:.4f} ctr={global_ctr:.4f} "
                    f"lr={global_lr:.4f} awt={global_awt:.4f} w={[round(x, 4) for x in w]}"
                )

    if diag_path and diag_writer:
        diag_f.close()
    if item_diag_path and item_diag_f:
        item_diag_f.close()
    if query_rank_path and query_rank_f:
        query_rank_f.close()

    outputs: dict[str, Path] = {"csv": csv_path}

    if not args.no_plots:
        try:
            import importlib

            matplotlib = importlib.import_module("matplotlib")
            matplotlib.use("Agg")
            plt = importlib.import_module("matplotlib.pyplot")
        except Exception:
            print("matplotlib is unavailable; skipping PNG generation.")
            return outputs

        rounds = list(range(1, args.rounds + 1))

        reward_png = out_dir / "reward_curve.png"
        plt.figure(figsize=(8, 4.5))
        plt.plot(rounds, rewards_history, color="#0f766e", linewidth=1.6)
        plt.title("Average Reward per Round")
        plt.xlabel("Round")
        plt.ylabel("Avg Reward")
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(reward_png, dpi=140)
        plt.close()
        outputs["reward_png"] = reward_png

        weights_png = out_dir / "weight_trajectory.png"
        plt.figure(figsize=(9, 5))
        labels = ["cos", "fresh", "lr", "ctr", "awt"]
        if use_social:
            labels.append("social")
        for i, label in enumerate(labels):
            series = [row[i] for row in weights_history]
            plt.plot(rounds, series, linewidth=1.5, label=label)
        plt.title("Weight Trajectory on Simplex")
        plt.xlabel("Round")
        plt.ylabel("Weight value")
        plt.ylim(0.0, 1.0)
        plt.grid(alpha=0.25)
        plt.legend(ncol=3)
        plt.tight_layout()
        plt.savefig(weights_png, dpi=140)
        plt.close()
        outputs["weights_png"] = weights_png

    return outputs


def main() -> None:
    parser = get_parser()
    args = parser.parse_args()

    outputs = run_simulation(args)
    print("\nSimulation finished.")
    print(f"CSV: {outputs['csv']}")
    if "reward_png" in outputs:
        print(f"Reward plot: {outputs['reward_png']}")
    if "weights_png" in outputs:
        print(f"Weight plot: {outputs['weights_png']}")


if __name__ == "__main__":
    main()
