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
    arrival_time: int = 0  # Round when image was injected (0 for initial images)
    last_shown_round: int = 0  # Last round when image was displayed to users

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

    parser.add_argument("--rho", type=float, default=0.2)
    parser.add_argument("--m-ctr", type=float, default=0.1)
    parser.add_argument("--m-lr", type=float, default=0.05)
    parser.add_argument("--c-ctr", type=float, default=10.0)
    parser.add_argument("--c-lr", type=float, default=10.0)
    parser.add_argument("--omega-l", type=float, default=5.0)
    parser.add_argument("--m-social", type=float, default=math.log1p(500.0))

    parser.add_argument("--gamma-pos", type=float, default=1.0)
    parser.add_argument("--kappa", type=float, default=10.0)

    parser.add_argument("--w-init", nargs=6, type=float, default=[1.0/6.0]*6, help="Initial weights [cos, fresh, lr, ctr, awt, social].")
    parser.add_argument("--eta", type=float, default=0.01)
    parser.add_argument("--tau", type=float, default=5.0)
    parser.add_argument("--beta", type=float, default=0.0)

    parser.add_argument("--lambda-v", type=float, default=1.0)
    parser.add_argument("--lambda-t", type=float, default=2.0)
    parser.add_argument("--lambda-l", type=float, default=5.0)
    parser.add_argument("--alpha", nargs=5, type=float, default=[-0.8, 0.6, 2.0, 0.3, 1.0], help="User behavior oracle alpha coefficients [intercept, match, cos, fresh, social].")
    parser.add_argument("--alpha-q", type=float, default=0.5, help="Quality coefficient for alpha.")
    parser.add_argument("--gamma", nargs=5, type=float, default=[-0.6, 0.4, 1.0, 0.2, 0.4], help="User behavior oracle gamma coefficients.")
    parser.add_argument("--gamma-q", type=float, default=2.5, help="Quality coefficient for gamma.")
    parser.add_argument("--delta", nargs=5, type=float, default=[-2.0, 0.3, 0.8, 0.1, 0.5], help="User behavior oracle delta coefficients.")
    parser.add_argument("--delta-t", type=float, default=2.0, help="Watch time coefficient for delta.")
    parser.add_argument("--delta-q", type=float, default=2.0, help="Quality coefficient for delta.")
    parser.add_argument("--user-random-sd", type=float, default=0.1, help="Standard deviation for user random effects.")
    parser.add_argument("--normalize-features", action="store_true", help="Min-max normalize ranking features to [0,1] before scoring.")
    parser.add_argument("--drop-social", action="store_true", help="Remove social_proof from the ranking feature vector.")
    parser.add_argument("--freeze-social", type=float, default=-1.0, help="If >=0, fix social weight to this value and optimize remaining dims on simplex.")
    parser.add_argument("--freeze-cos", type=float, default=-1.0, help="If >=0, fix cosine weight to this value and optimize remaining dims on simplex.")
    parser.add_argument("--lagged-social", action="store_true", help="Use previous-round social proof instead of current-round.")

    parser.add_argument("--out-dir", type=str, default="backend/data/simulation")
    parser.add_argument(
        "--dataset-cache",
        type=str,
        default="",
        help="Path to dataset cosine cache JSON. If set, simulation uses cache-derived image-query cosines.",
    )
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--item-diagnostics", action="store_true", help="Write per-image stats and rank trajectories to item_diagnostics.csv.")
    parser.add_argument("--two-stage-top-k", type=int, default=10, help="If >0, use two-stage ranking: Stage 1 filters by cosine to top-K, Stage 2 re-ranks by quality_score.")
    parser.add_argument("--quality-weight-cos", type=float, default=1.0, help="Weight for cosine in stage 2 quality_score.")
    parser.add_argument("--quality-weight-match", type=float, default=2.0, help="Weight for category match in stage 2 quality_score.")
    parser.add_argument("--quality-weight-ctr", type=float, default=1.0, help="Weight for CTR in stage 2 quality_score.")
    parser.add_argument("--quality-weight-lr", type=float, default=1.0, help="Weight for LR in stage 2 quality_score.")
    parser.add_argument("--quality-weight-awt", type=float, default=1.0, help="Weight for AWT in stage 2 quality_score.")
    parser.add_argument("--quality-weight-fresh", type=float, default=0.5, help="Weight for freshness in stage 2 quality_score.")
    parser.add_argument("--no-simplex-projection", action="store_true", help="Disable simplex projection on weight updates.")
    parser.add_argument("--inject-images", action="store_true", help="Enable three-phase injection experiment.")
    parser.add_argument("--inject-round", type=int, default=501, help="Round at which to inject new images.")
    parser.add_argument("--inject-category", type=str, default="cat", help="Category to inject images into.")
    parser.add_argument("--min-fresh-weight", type=float, default=0.2, help="Minimum weight for freshness after each update.")
    parser.add_argument("--freshness-decay-scale", type=float, default=5.0, help="Scale parameter for freshness decay: f = 2^(-(t - t_arrival) / scale).")
    parser.add_argument("--recovery-boost-interval", type=int, default=50, help="Every N rounds, apply recovery boost to images not shown recently.")
    parser.add_argument("--recovery-boost-strength", type=float, default=0.5, help="Temporary freshness boost for images not shown in last N rounds.")
    return parser


def run_simulation(args: argparse.Namespace) -> dict[str, Path]:
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
    alpha = args.alpha
    gamma = args.gamma
    delta = args.delta
    alpha_q = args.alpha_q
    gamma_q = args.gamma_q
    delta_t = args.delta_t
    delta_q = args.delta_q

    # w over x=[cos, freshness, LR~, CTR~, AWT~, social_proof]
    w = args.w_init[:]
    dim = 6 - (1 if args.drop_social else 0)
    if args.freeze_social >= 0:
        w[5] = args.freeze_social
    if args.freeze_cos >= 0:
        w[0] = args.freeze_cos

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "simulation_rounds.csv"

    item_diag_path = out_dir / "item_stats.csv" if args.item_diagnostics else None
    item_diag_f = None
    item_diag_writer = None
    if item_diag_path:
        item_diag_f = item_diag_path.open("w", newline="", encoding="utf-8")
        item_diag_writer = csv.writer(item_diag_f)
        item_diag_writer.writerow(["round", "image_id", "category", "quality", "freshness", "ctr", "lr", "awt", "social_proof", "cos_ui_own", "own_query_score"])

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

    rewards_history: list[float] = []
    weights_history: list[list[float]] = []

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
            # Update freshness with decay formula: f = 2^(-(t - t_arrival) / scale)
            if args.inject_images:
                for img in images:
                    age = t - img.arrival_time
                    img.freshness = 2.0 ** (-age / args.freshness_decay_scale)
            
            # Recovery Boost: Every N rounds, give unshown images a temporary freshness boost
            if args.recovery_boost_interval > 0 and t % args.recovery_boost_interval == 0:
                for img in images:
                    rounds_since_shown = t - img.last_shown_round
                    if rounds_since_shown >= args.recovery_boost_interval:
                        # Image hasn't been shown for N+ rounds, give recovery boost
                        img.freshness = min(1.0, img.freshness + args.recovery_boost_strength)
                if t % 100 == 0:  # Log every 100 rounds
                    print(f"Round {t}: Applied recovery boost to images not shown in last {args.recovery_boost_interval} rounds")
            
            # Inject new images at specified round
            if args.inject_images and t == args.inject_round:
                inject_category_idx = query_labels.index(args.inject_category) if args.inject_category in query_labels else 0
                # Compute average cosine of existing images in that category for realism
                cat_cosines = [
                    float((img.cosine_by_query or {}).get(args.inject_category, 0.0))
                    for img in images if img.category == inject_category_idx
                ]
                avg_cat_cos = sum(cat_cosines) / len(cat_cosines) if cat_cosines else 0.1
                # Assign cosine values proportional to quality
                cos_high = min(1.0, avg_cat_cos + 0.02)
                cos_mid  = min(1.0, avg_cat_cos + 0.01)  # slight edge to enter top-10
                cos_low  = max(0.0, avg_cat_cos - 0.02)
                print(f"Round {t}: Injecting with avg category cosine={avg_cat_cos:.4f}")

                def _seed_engagement(img: ImageState, virtual_exams: float, p_ctr: float, p_lr: float, awt_val: float) -> None:
                    """Seed quality-proportional synthetic engagement to break cold-start deadlock."""
                    img.tilde_e = virtual_exams
                    img.tilde_v = virtual_exams * p_ctr
                    img.tilde_l = img.tilde_v * p_lr
                    img.ctr = (img.tilde_v + args.c_ctr * args.m_ctr) / (img.tilde_e + args.c_ctr)
                    img.lr  = (img.tilde_l + args.c_lr  * args.m_lr)  / (img.tilde_v + args.c_lr)
                    img.awt = awt_val

                # Inject high quality image (q=0.4)
                img_high = ImageState(
                    image_id=len(images), category=inject_category_idx,
                    embedding=[0.0] * args.embedding_dim, quality=0.4, freshness=1.0,
                    arrival_time=t,
                    cosine_by_query={args.inject_category: cos_high} if use_dataset_cache else None
                )
                _seed_engagement(img_high, virtual_exams=50, p_ctr=0.28, p_lr=0.25, awt_val=0.45)
                images.append(img_high)

                # Inject mid quality image (q=0.25)
                img_mid = ImageState(
                    image_id=len(images), category=inject_category_idx,
                    embedding=[0.0] * args.embedding_dim, quality=0.25, freshness=1.0,
                    arrival_time=t,
                    cosine_by_query={args.inject_category: cos_mid} if use_dataset_cache else None
                )
                _seed_engagement(img_mid, virtual_exams=50, p_ctr=0.20, p_lr=0.17, awt_val=0.30)
                images.append(img_mid)

                # Inject low quality image (q=0.01)
                img_low = ImageState(
                    image_id=len(images), category=inject_category_idx,
                    embedding=[0.0] * args.embedding_dim, quality=0.01, freshness=1.0,
                    arrival_time=t,
                    cosine_by_query={args.inject_category: cos_low} if use_dataset_cache else None
                )
                _seed_engagement(img_low, virtual_exams=50, p_ctr=0.06, p_lr=0.03, awt_val=0.05)
                images.append(img_low)

                print(f"Round {t}: Injected 3 images into category '{args.inject_category}' (high q=0.4 cos={cos_high:.4f}, mid q=0.25 cos={cos_mid:.4f}, low q=0.01 cos={cos_low:.4f}) with seeded engagement")
            
            prev_social = {img.image_id: img.social_proof for img in images} if args.lagged_social else {}
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

            grad = [0.0] * 6
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
                    if not args.drop_social:
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

                # Two-stage ranking: Stage 1 filters by cosine, Stage 2 re-ranks by quality_score
                if args.two_stage_top_k > 0:
                    # Stage 1: Filter by cosine to top-K
                    stage1_ranked = sorted(range(len(images)), key=lambda i: cos_values[i], reverse=True)
                    stage1_top_k = stage1_ranked[: args.two_stage_top_k]
                    
                    # Stage 2: Re-rank top-K by quality_score = w_cos*cos + w_match*match + w_fresh*fresh + w_ctr*CTR + w_lr*LR + w_awt*AWT
                    stage2_scores = []
                    for idx in stage1_top_k:
                        img = images[idx]
                        match = 1.0 if img.category == query_cat else 0.0
                        quality_score = (
                            args.quality_weight_cos * cos_values[idx] +
                            args.quality_weight_match * match +
                            args.quality_weight_fresh * img.freshness +
                            args.quality_weight_ctr * img.ctr +
                            args.quality_weight_lr * img.lr +
                            args.quality_weight_awt * img.awt
                        )
                        stage2_scores.append(quality_score)
                    
                    # Re-rank stage1_top_k by stage2_scores
                    stage2_ranked = sorted(range(len(stage1_top_k)), key=lambda i: stage2_scores[i], reverse=True)
                    displayed = [stage1_top_k[i] for i in stage2_ranked[: args.top_k]]
                    
                    # Exploration Injection: 5% chance to replace 10th image with random non-top-10 image
                    if len(displayed) >= args.top_k and rng.random() < 0.05:
                        # Get pool of images not in top 10 (from stage1_top_k)
                        non_displayed_pool = [idx for idx in stage1_top_k if idx not in displayed]
                        if non_displayed_pool:
                            random_idx = rng.choice(non_displayed_pool)
                            # Replace 10th ranked image (index -1 in displayed) with random image
                            displayed[-1] = random_idx
                    
                    # For policy gradient, use stage2 scores for displayed items, 0 for others
                    policy_scores = [0.0] * len(images)
                    for i, idx in enumerate(stage1_top_k):
                        policy_scores[idx] = stage2_scores[i]
                else:
                    ranked_indices = sorted(range(len(images)), key=lambda i: scores[i], reverse=True)
                    displayed = ranked_indices[: args.top_k]
                    policy_scores = scores

                # Policy over all candidates (reward of non-displayed defaults to 0).
                logits = [args.tau * s for s in policy_scores]
                a = softmax(logits)
                rewards = [0.0] * len(images)

                for rank, idx in enumerate(displayed):
                    img = images[idx]
                    # Update last shown round for recovery boost tracking
                    img.last_shown_round = t
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
                    
                    # Penalty for Non-Clicks: immediate CTR update if examined but not clicked
                    if view == 0.0:
                        # Immediately update CTR to reflect failure (penalize score)
                        agg = image_aggr[img.image_id]
                        e_t = agg["e"]
                        v_t = agg["v"]
                        if e_t > 0:
                            # Apply Bayesian smoothing to get current CTR
                            current_ctr = (v_t + args.c_ctr * args.m_ctr) / (e_t + args.c_ctr)
                            img.ctr = current_ctr

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
                # Only calculate gradient for displayed (top-K) images
                for i in displayed:
                    coeff = -args.tau * a[i] * (rewards[i] - expected_reward)
                    x = x_vectors[i]
                    for j in range(6):
                        grad[j] += coeff * x[j]

            batch_size = max(len(batch_users), 1)
            grad = [g / batch_size + 2.0 * args.beta * wj for g, wj in zip(grad, w)]
            if args.no_simplex_projection:
                w = [wj - args.eta * gj for wj, gj in zip(w, grad)]
            else:
                w = project_to_simplex([wj - args.eta * gj for wj, gj in zip(w, grad)])
            
            # Apply weight constraints
            w = [max(wj, 0.0) for wj in w]  # Clip to non-negative
            if args.min_fresh_weight > 0:
                w[1] = max(w[1], args.min_fresh_weight)  # Force w_fresh >= min_fresh_weight

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
                    "w5_social": f"{w[5]:.6f}",
                }
            )

            rewards_history.append(avg_reward)
            weights_history.append(w[:])

            if t % 50 == 0 or t == 1 or t == args.rounds:
                print(
                    f"round={t:4d} reward={avg_reward:.4f} ctr={global_ctr:.4f} "
                    f"lr={global_lr:.4f} awt={global_awt:.4f} w={[round(x, 4) for x in w]}"
                )

            # Write per-image stats with own-query score
            if item_diag_writer and (t % 10 == 0 or t == 1 or t == args.rounds):
                tracked_queries = query_labels if use_dataset_cache else [f"cat_{i}" for i in range(args.num_categories)]
                for img in images:
                    if use_dataset_cache:
                        own_label = tracked_queries[img.category] if img.category < len(tracked_queries) else tracked_queries[0]
                        cos_ui = float((img.cosine_by_query or {}).get(own_label, 0.0))
                    else:
                        cos_ui = 0.0
                    x = [cos_ui, img.freshness, img.lr, img.ctr, img.awt]
                    if not args.drop_social:
                        x.append(img.social_proof)
                    own_score = dot(w, x)
                    item_diag_writer.writerow([t, img.image_id, img.category, f"{img.quality:.6f}", f"{img.freshness:.6f}", f"{img.ctr:.6f}", f"{img.lr:.6f}", f"{img.awt:.6f}", f"{img.social_proof:.6f}", f"{cos_ui:.6f}", f"{own_score:.6f}"])

            # Write query rankings
            if query_rank_writer and (t % 10 == 0 or t == 1 or t == args.rounds):
                tracked_queries = query_labels if use_dataset_cache else [f"cat_{i}" for i in range(args.num_categories)]
                for q_label in tracked_queries:
                    # Get cosine values for this query
                    cos_values_q = []
                    for img in images:
                        if use_dataset_cache:
                            cos_ui = float((img.cosine_by_query or {}).get(q_label, 0.0))
                        else:
                            cos_ui = 0.0
                        cos_values_q.append(cos_ui)
                    
                    # Two-stage ranking
                    if args.two_stage_top_k > 0:
                        # Stage 1: Filter by cosine to top-K
                        stage1_ranked = sorted(range(len(images)), key=lambda i: cos_values_q[i], reverse=True)
                        stage1_top_k = stage1_ranked[: args.two_stage_top_k]
                        
                        # Stage 2: Re-rank by quality_score
                        stage2_scores = []
                        for idx in stage1_top_k:
                            img = images[idx]
                            match = 1.0 if img.category == query_labels.index(q_label) else 0.0
                            quality_score = (
                                args.quality_weight_cos * cos_values_q[idx] +
                                args.quality_weight_match * match +
                                args.quality_weight_fresh * img.freshness +
                                args.quality_weight_ctr * img.ctr +
                                args.quality_weight_lr * img.lr +
                                args.quality_weight_awt * img.awt
                            )
                            stage2_scores.append(quality_score)
                        
                        # Re-rank stage1_top_k by stage2_scores
                        stage2_ranked = sorted(range(len(stage1_top_k)), key=lambda i: stage2_scores[i], reverse=True)
                        displayed = [stage1_top_k[i] for i in stage2_ranked[: args.top_k]]
                        
                        # Write ALL stage1 items with their full rank (including those outside display top-k)
                        for rank, i in enumerate(stage2_ranked, start=1):
                            idx = stage1_top_k[i]
                            img = images[idx]
                            query_rank_writer.writerow([t, q_label, img.image_id, img.category, rank, f"{stage2_scores[i]:.6f}"])
                    else:
                        # Standard ranking by learned weights
                        scores_q: list[tuple[float, int, int]] = []
                        for img in images:
                            cos_ui = cos_values_q[img.image_id]
                            x = [cos_ui, img.freshness, img.lr, img.ctr, img.awt]
                            if not args.drop_social:
                                x.append(img.social_proof)
                            s = dot(w, x)
                            scores_q.append((s, img.image_id, img.category))
                        scores_q.sort(key=lambda x: x[0], reverse=True)
                        for rank, (s, iid, cat) in enumerate(scores_q, start=1):
                            query_rank_writer.writerow([t, q_label, iid, cat, rank, f"{s:.6f}"])

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
        labels = ["cos", "fresh", "lr", "ctr", "awt", "social"]
        for i, label in enumerate(labels):
            series = [row[i] for row in weights_history]
            plt.plot(rounds, series, linewidth=1.5, label=label)
        plt.title("Weight Trajectory on Simplex")
        plt.xlabel("Round")
        plt.ylabel("Weight value")
        if not args.no_simplex_projection:
            plt.ylim(0.0, 1.0)
        plt.grid(alpha=0.25)
        plt.legend(ncol=3)
        plt.tight_layout()
        plt.savefig(weights_png, dpi=140)
        plt.close()
        outputs["weights_png"] = weights_png

    if item_diag_f:
        item_diag_f.close()
    if query_rank_f:
        query_rank_f.close()

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
