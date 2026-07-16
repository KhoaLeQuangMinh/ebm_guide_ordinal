from pathlib import Path

import numpy as np
from scipy.stats import norm

from pySuStaIn.MixedTypeSustain import MixedTypeSustain


def create_score_vals(n_biomarkers, list_scores):
    if n_biomarkers == 0:
        if len(list_scores) != 0:
            raise ValueError("list_scores must be empty when n_biomarkers is 0")
        return np.zeros((0, 0), dtype=int)

    if len(list_scores) != n_biomarkers:
        raise ValueError(
            f"list_scores length {len(list_scores)} does not match n_biomarkers {n_biomarkers}"
        )
    max_score = int(np.max(list_scores))
    score_vals = np.zeros((n_biomarkers, max_score), dtype=int)
    for i, score in enumerate(list_scores):
        score_vals[i, :score] = np.arange(1, score + 1)
    return score_vals


def create_distribution(score, prob, dist_type, ind):
    if dist_type == "categorical":
        dist = np.full((score + 1), (1 - prob) / score)
        dist[ind] = prob
        return dist

    if dist_type == "gaussian":
        std_dev = 1
        mean_index = ind
        indices = np.arange(score + 1)
        gaussian_probs = np.exp(-0.5 * ((indices - mean_index) / std_dev) ** 2)
        gaussian_probs[ind] = prob
        remaining_sum = np.sum(gaussian_probs) - prob
        dist = gaussian_probs * (1 - prob) / remaining_sum
        dist[ind] = prob
        return dist

    raise ValueError(f"Unknown dist_type: {dist_type}")


def create_prob_nl(data, list_scores, prob_correct):
    if data is None or data.shape[1] == 0:
        return np.zeros((data.shape[0], 0)) if data is not None else None
    n_subjects, n_biomarkers = data.shape
    prob_nl = np.zeros((n_subjects, n_biomarkers))
    for i, score in enumerate(list_scores):
        prob = prob_correct[i]
        p_nl_dist = create_distribution(score, prob, dist_type="gaussian", ind=0)
        prob_nl[:, i] = p_nl_dist[data[:, i].astype(int)]
    return prob_nl


def create_prob_score(data, list_scores, prob_correct):
    if data is None or data.shape[1] == 0:
        return np.zeros((data.shape[0], 0, 0)) if data is not None else None
    n_subjects, n_biomarkers = data.shape
    max_score = int(np.max(list_scores))
    prob_score = np.zeros((n_subjects, n_biomarkers, max_score))

    p_score_dist_list = []
    for i, score in enumerate(list_scores):
        p_score_dist = np.full((score, score + 1), (1 - prob_correct[i]) / score)
        for j in range(score):
            dist = create_distribution(score, prob_correct[i], dist_type="gaussian", ind=j + 1)
            p_score_dist[j, :] = dist
        p_score_dist_list.append(p_score_dist)

    for n in range(n_biomarkers):
        score = list_scores[n]
        for z in range(score):
            for score_value in range(score + 1):
                prob_score[data[:, n] == score_value, n, z] = p_score_dist_list[n][z, score_value]

    return prob_score


def generate_random_sequence(scores, n_subtypes, seed):
    np.random.seed(seed)
    num_biomarkers = scores.shape[0]
    max_score = scores.shape[1]

    stage_score = scores.T.flatten()
    ix_select = stage_score > 0
    stage_score = stage_score[ix_select]

    stage_biomarker_index = np.tile(np.arange(num_biomarkers), (max_score,))
    stage_biomarker_index = stage_biomarker_index[ix_select]

    num_stages = stage_score.shape[0]
    seq = np.zeros((n_subtypes, num_stages))

    possible_biomarkers = np.unique(stage_biomarker_index)

    for s in range(n_subtypes):
        for i in range(num_stages):
            is_min_stage_score = np.full(num_stages, False)

            for b in possible_biomarkers:
                is_unselected = np.full(num_stages, False)
                for k in set(range(num_stages)) - set(seq[s][:i]):
                    is_unselected[k] = True

                this_biomarkers = np.logical_and(
                    stage_biomarker_index == b,
                    np.array(is_unselected) == 1
                )
                if not np.any(this_biomarkers):
                    this_min_stage_score = 0
                else:
                    this_min_stage_score = np.min(stage_score[this_biomarkers])

                if this_min_stage_score:
                    is_min_stage_score[np.logical_and(
                        this_biomarkers,
                        stage_score == this_min_stage_score
                    )] = True

            events = np.arange(num_stages)
            possible_events = events[is_min_stage_score]
            this_index = np.ceil(np.random.rand() * len(possible_events)) - 1
            seq[s][i] = possible_events[int(this_index)]

    return seq


def generate_zscore_mixed_data(
    n_samples,
    n_subtypes,
    n_biomarkers_zscore,
    z_max,
    gt_subtypes,
    gt_stages,
    gt_sequence,
    stage_biomarker_index,
    stage_score,
    bool_zscore_biomarkers,
    n_biomarkers_total,
):
    n_stages = gt_sequence.shape[1]
    zscore_data = np.zeros((n_samples, n_biomarkers_zscore))

    min_biomarker_zscore = [0] * n_biomarkers_zscore
    max_biomarker_zscore = z_max
    std_biomarker_zscore = 1

    stage_value = np.zeros((n_biomarkers_zscore, n_stages + 2, n_subtypes))

    for s in range(n_subtypes):
        seq = gt_sequence[s, :]
        seq_inv = np.array([0] * n_stages)
        seq_inv[seq.astype(int)] = np.arange(n_stages)

        idx_zscore = 0
        for b in range(n_biomarkers_total):
            if bool_zscore_biomarkers[b]:
                event_location = np.concatenate([[0], seq_inv[(stage_biomarker_index == b)], [n_stages]])
                event_value = np.concatenate(
                    [[min_biomarker_zscore[idx_zscore]], stage_score[stage_biomarker_index == b],
                     [max_biomarker_zscore[idx_zscore]]]
                )

                for j in range(len(event_location) - 1):
                    if j == 0:
                        index = np.arange(event_location[j], event_location[j + 1] + 2)
                        stage_value[idx_zscore, index, s] = np.linspace(
                            event_value[j], event_value[j + 1], event_location[j + 1] - event_location[j] + 2
                        )
                    else:
                        index = np.arange(event_location[j] + 1, event_location[j + 1] + 2)
                        stage_value[idx_zscore, index, s] = np.linspace(
                            event_value[j], event_value[j + 1], event_location[j + 1] - event_location[j] + 1
                        )

                idx_zscore += 1

    for m in range(n_samples):
        idx_zscore = 0
        for b in range(n_biomarkers_total):
            if bool_zscore_biomarkers[b]:
                zscore_data[m, idx_zscore] = stage_value[idx_zscore, int(gt_stages[m]), gt_subtypes[m]]
                idx_zscore += 1

    zscore_data = zscore_data + norm.ppf(np.random.rand(n_biomarkers_zscore, n_samples).T) * np.tile(
        std_biomarker_zscore, (n_samples, 1)
    )

    return zscore_data


def generate_ordinal_mixed_data(
    n_samples,
    n_biomarkers_ordinal,
    n_subtypes,
    prob_correct,
    ordinal_scores,
    gt_subtypes,
    gt_stages,
    gt_sequence,
    stage_biomarker_index,
    stage_score,
    bool_ordinal_biomarkers,
):
    ordinal_data = np.zeros((n_samples, n_biomarkers_ordinal))
    n_stages = gt_sequence.shape[1]
    n_biomarkers_total = len(bool_ordinal_biomarkers)
    stage_value_ordinal = np.zeros((n_stages + 1, n_biomarkers_total, n_subtypes))

    for s in range(n_subtypes):
        seq = gt_sequence[s, :]

        for stage in range(n_stages):
            index_justreached = int(seq[stage])
            biomarker_justreached = int(stage_biomarker_index[index_justreached])
            stage_value_justreached = int(stage_score[index_justreached])

            if bool_ordinal_biomarkers[biomarker_justreached]:
                index = stage + 1
                stage_value_ordinal[index:, biomarker_justreached, s] = stage_value_justreached

    for b in range(n_biomarkers_ordinal):
        prob = prob_correct[b]
        score = ordinal_scores[b]
        p_nl_dist = create_distribution(score, prob, dist_type="categorical", ind=0)
        ordinal_data[:, b] = np.random.choice(score + 1, n_samples, replace=True, p=p_nl_dist)

    for m in range(n_samples):
        this_subtype = gt_subtypes[m]
        this_stage = gt_stages[m, 0].astype(int)

        stage_value_justreached = stage_value_ordinal[this_stage, :, this_subtype]
        stage_value_justreached = stage_value_justreached[bool_ordinal_biomarkers]

        for i, value in enumerate(stage_value_justreached):
            if value > 0:
                idx_value = value.astype(int)
                idx_biomarker = i

                prob_correct_biomarker = prob_correct[idx_biomarker]
                p_score_dist = create_distribution(
                    ordinal_scores[idx_biomarker], prob_correct_biomarker, dist_type="categorical", ind=idx_value
                )

                ordinal_data[m, idx_biomarker] = np.random.choice(
                    ordinal_scores[idx_biomarker] + 1, 1, replace=True, p=p_score_dist
                )

    return ordinal_data


def generate_event_mixed_data(
    n_samples,
    n_biomarkers_event,
    n_subtypes,
    gt_subtypes,
    gt_stages,
    gt_sequence,
    stage_biomarker_index,
    bool_event_biomarkers,
    mixture_style="mixture_GMM",
):
    if n_biomarkers_event == 0:
        return None, None, None

    if mixture_style not in ("mixture_GMM",):
        raise NotImplementedError("Only mixture_GMM event simulation is currently supported in simfuncs_mixed.")

    n_stages = gt_sequence.shape[1]
    n_biomarkers_total = len(bool_event_biomarkers)
    event_global_idx = np.where(bool_event_biomarkers)[0]

    stage_event_reached = np.zeros((n_stages + 1, n_biomarkers_total, n_subtypes), dtype=bool)
    for s in range(n_subtypes):
        S = gt_sequence[s, :]
        for stage in range(n_stages):
            index_justreached = int(S[stage])
            biomarker_justreached = int(stage_biomarker_index[index_justreached])
            if bool_event_biomarkers[biomarker_justreached]:
                stage_event_reached[stage + 1:, biomarker_justreached, s] = True

    mean_controls = np.array([0.0] * n_biomarkers_event)
    std_controls = np.array([0.25] * n_biomarkers_event)
    mean_cases = np.array([1.5] * n_biomarkers_event)
    std_cases = np.random.uniform(0.25, 0.50, n_biomarkers_event)

    event_data = np.zeros((n_samples, n_biomarkers_event))
    for m in range(n_samples):
        subtype_m = gt_subtypes[m]
        stage_m = int(gt_stages[m, 0])
        reached_global = stage_event_reached[stage_m, :, subtype_m]
        reached_local = reached_global[event_global_idx]

        for b in range(n_biomarkers_event):
            if reached_local[b]:
                event_data[m, b] = np.random.normal(mean_cases[b], std_cases[b])
            else:
                event_data[m, b] = np.random.normal(mean_controls[b], std_controls[b])

    event_prob_no = np.zeros((n_samples, n_biomarkers_event))
    event_prob_yes = np.zeros((n_samples, n_biomarkers_event))
    for b in range(n_biomarkers_event):
        control_pdf = norm.pdf(event_data[:, b], loc=mean_controls[b], scale=std_controls[b])
        case_pdf = norm.pdf(event_data[:, b], loc=mean_cases[b], scale=std_cases[b])
        normalizer = control_pdf + case_pdf + 1e-250
        event_prob_no[:, b] = control_pdf / normalizer
        event_prob_yes[:, b] = case_pdf / normalizer

    return event_data, event_prob_yes, event_prob_no


def generate_mixed_data(
    seed=42,
    n_samples=500,
    n_subtypes=2,
    n_biomarkers_zscore=4,
    n_biomarkers_ordinal=0,
    n_biomarkers_event=0,
    z_vals=None,
    z_max=None,
    list_scores=None,
    prob_correct=None,
    mixture_style="mixture_GMM",
):
    np.random.seed(seed)

    if n_biomarkers_zscore < 0 or n_biomarkers_ordinal < 0 or n_biomarkers_event < 0:
        raise ValueError("n_biomarkers_* values must be >= 0")

    if n_biomarkers_zscore == 0:
        if z_vals is None:
            z_vals = np.zeros((0, 0), dtype=int)
        else:
            z_vals = np.asarray(z_vals)
            if z_vals.shape[0] != 0:
                raise ValueError("z_vals must have 0 rows when n_biomarkers_zscore is 0")

        if z_max is None:
            z_max = np.array([], dtype=int)
        else:
            z_max = np.asarray(z_max)
            if z_max.shape[0] != 0:
                raise ValueError("z_max must be empty when n_biomarkers_zscore is 0")
    else:
        if z_vals is None:
            z_vals = np.tile(np.array([1, 2, 3]), (n_biomarkers_zscore, 1))
        else:
            z_vals = np.asarray(z_vals)
            if z_vals.shape[0] != n_biomarkers_zscore:
                raise ValueError(
                    f"z_vals has {z_vals.shape[0]} rows but n_biomarkers_zscore is {n_biomarkers_zscore}"
                )

        if z_max is None:
            z_max = np.array([5] * n_biomarkers_zscore)
        else:
            z_max = np.asarray(z_max)
            if z_max.shape[0] != n_biomarkers_zscore:
                raise ValueError(
                    f"z_max length {z_max.shape[0]} does not match n_biomarkers_zscore {n_biomarkers_zscore}"
                )

    n_biomarkers_ordinal_total = n_biomarkers_ordinal + n_biomarkers_event
    if n_biomarkers_ordinal_total > 0:
        if list_scores is None:
            ordinal_max_scores = np.array([3] * n_biomarkers_ordinal)
            event_max_scores = np.array([1] * n_biomarkers_event)
            list_scores = np.concatenate([ordinal_max_scores, event_max_scores])
        else:
            list_scores = np.asarray(list_scores)
            if list_scores.shape[0] != n_biomarkers_ordinal_total:
                raise ValueError(
                    "list_scores length does not match n_biomarkers_ordinal + n_biomarkers_event"
                )

        if prob_correct is None:
            prob_correct = np.concatenate([
                np.full(n_biomarkers_ordinal, 0.70),
                np.full(n_biomarkers_event, 0.75),
            ])
        else:
            prob_correct = np.asarray(prob_correct)
            if prob_correct.shape[0] != n_biomarkers_ordinal_total:
                raise ValueError(
                    "prob_correct length does not match n_biomarkers_ordinal + n_biomarkers_event"
                )

        score_vals = create_score_vals(n_biomarkers_ordinal_total, list_scores)
    else:
        if list_scores is not None and len(list_scores) != 0:
            raise ValueError("list_scores must be empty when there are no ordinal/event biomarkers")
        if prob_correct is not None and len(prob_correct) != 0:
            raise ValueError("prob_correct must be empty when there are no ordinal/event biomarkers")
        list_scores = np.array([])
        prob_correct = np.array([])
        score_vals = np.zeros((0, 0), dtype=int)

    n_biomarkers_total = n_biomarkers_zscore + n_biomarkers_ordinal + n_biomarkers_event
    max_score_value = max(z_vals.shape[1] if z_vals.size else 0, score_vals.shape[1] if score_vals.size else 0)

    mixed_data_vals = np.zeros((n_biomarkers_total, max_score_value))

    for i in range(n_biomarkers_zscore):
        mixed_data_vals[i, : z_vals.shape[1]] = z_vals[i]
    for i in range(n_biomarkers_ordinal + n_biomarkers_event):
        mixed_data_vals[n_biomarkers_zscore + i, : score_vals.shape[1]] = score_vals[i]

    bool_zscore_biomarkers = np.array(
        [True] * n_biomarkers_zscore
        + [False] * (n_biomarkers_ordinal + n_biomarkers_event)
    )
    bool_ordinal_biomarkers = np.array(
        [False] * n_biomarkers_zscore
        + [True] * (n_biomarkers_ordinal + n_biomarkers_event)
    )
    bool_ordinal_only_biomarkers = np.array(
        [False] * n_biomarkers_zscore
        + [True] * n_biomarkers_ordinal
        + [False] * n_biomarkers_event
    )
    bool_event_biomarkers = np.array(
        [False] * (n_biomarkers_zscore + n_biomarkers_ordinal)
        + [True] * n_biomarkers_event
    )

    gt_sequence = generate_random_sequence(mixed_data_vals, n_subtypes, seed=seed)
    gt_f = [1 + 0.5 * x for x in range(n_subtypes)]
    gt_f = [x / sum(gt_f) for x in gt_f][::-1]
    gt_subtypes = np.random.default_rng(seed).choice(
        range(n_subtypes), n_samples, replace=True, p=gt_f
    ).astype(int)

    n_stages = np.sum(mixed_data_vals > 0)
    n_k = n_stages + 1
    gt_stages = np.ceil(np.random.rand(n_samples, 1) * n_k) - 1

    stage_score = mixed_data_vals.T.flatten()
    ix_select = stage_score > 0
    stage_score = stage_score[ix_select]
    stage_biomarker_index = np.tile(np.arange(n_biomarkers_total), (max_score_value,))
    stage_biomarker_index = stage_biomarker_index[ix_select]

    zscore_data = generate_zscore_mixed_data(
        n_samples=n_samples,
        n_subtypes=n_subtypes,
        n_biomarkers_zscore=n_biomarkers_zscore,
        z_max=z_max,
        gt_subtypes=gt_subtypes,
        gt_stages=gt_stages,
        gt_sequence=gt_sequence,
        stage_biomarker_index=stage_biomarker_index,
        stage_score=stage_score,
        bool_zscore_biomarkers=bool_zscore_biomarkers,
        n_biomarkers_total=n_biomarkers_total,
    )

    ordinal_only_data = None
    event_data = None
    ordinal_prob_nl = None
    ordinal_prob_score = None
    event_prob_yes = None
    event_prob_no = None

    if n_biomarkers_ordinal > 0:
        ordinal_data = generate_ordinal_mixed_data(
            n_samples=n_samples,
            n_biomarkers_ordinal=n_biomarkers_ordinal,
            n_subtypes=n_subtypes,
            prob_correct=prob_correct[:n_biomarkers_ordinal],
            ordinal_scores=list_scores[:n_biomarkers_ordinal],
            gt_subtypes=gt_subtypes,
            gt_stages=gt_stages,
            gt_sequence=gt_sequence,
            stage_biomarker_index=stage_biomarker_index,
            stage_score=stage_score,
            bool_ordinal_biomarkers=bool_ordinal_only_biomarkers,
        )
        ordinal_only_data = ordinal_data
        ordinal_prob_nl = create_prob_nl(ordinal_only_data, list_scores[:n_biomarkers_ordinal], prob_correct[:n_biomarkers_ordinal])
        ordinal_prob_score = create_prob_score(ordinal_only_data, list_scores[:n_biomarkers_ordinal], prob_correct[:n_biomarkers_ordinal])
    else:
        ordinal_data = None

    if n_biomarkers_event > 0:
        event_data, event_prob_yes, event_prob_no = generate_event_mixed_data(
            n_samples=n_samples,
            n_biomarkers_event=n_biomarkers_event,
            n_subtypes=n_subtypes,
            gt_subtypes=gt_subtypes,
            gt_stages=gt_stages,
            gt_sequence=gt_sequence,
            stage_biomarker_index=stage_biomarker_index,
            bool_event_biomarkers=bool_event_biomarkers,
            mixture_style=mixture_style,
        )

    return {
        "zscore_data": zscore_data,
        "ordinal_data": ordinal_data,
        "ordinal_only_data": ordinal_only_data,
        "event_data": event_data,
        "ordinal_prob_nl": ordinal_prob_nl,
        "ordinal_prob_score": ordinal_prob_score,
        "event_prob_yes": event_prob_yes,
        "event_prob_no": event_prob_no,
        "z_vals": z_vals,
        "z_max": z_max,
        "score_vals": score_vals,
        "list_scores": list_scores,
        "prob_correct": prob_correct,
        "n_biomarkers_ordinal": n_biomarkers_ordinal,
        "n_biomarkers_event": n_biomarkers_event,
        "gt_sequence": gt_sequence,
        "gt_subtypes": gt_subtypes,
        "gt_stages": gt_stages,
    }


def main():
    seed = 42
    data = generate_mixed_data(seed=seed)

    zscore_data = data["zscore_data"]
    ordinal_data = data["ordinal_data"]
    z_vals = data["z_vals"]
    z_max = data["z_max"]
    score_vals = data["score_vals"]
    list_scores = data["list_scores"]
    prob_correct = data["prob_correct"]
    n_biomarkers_ordinal = data["n_biomarkers_ordinal"]
    n_biomarkers_event = data["n_biomarkers_event"]

    ordinal_prob_nl = data.get("ordinal_prob_nl")
    ordinal_prob_score = data.get("ordinal_prob_score")
    event_prob_yes = data.get("event_prob_yes")
    event_prob_no = data.get("event_prob_no")

    zscore_labels = [f"zscore_{i+1}" for i in range(zscore_data.shape[1])]
    ordinal_labels = [f"ordinal_{i+1}" for i in range(n_biomarkers_ordinal)]
    event_labels = [f"event_{i+1}" for i in range(n_biomarkers_event)]

    if ordinal_prob_nl is not None and n_biomarkers_ordinal > 0:
        ordinal_score_vals = score_vals[:n_biomarkers_ordinal, :]
    else:
        ordinal_prob_nl = None
        ordinal_prob_score = None
        ordinal_score_vals = []

    if event_prob_no is None or n_biomarkers_event == 0:
        event_prob_no = None
        event_prob_yes = None

    N_startpoints = 10
    N_S_max = 3
    N_iterations_MCMC = int(1e4)
    use_parallel_startpoints = True
    dataset_name = "sim_mixed"
    output_folder = Path.cwd() / f"{dataset_name}_output"
    output_folder.mkdir(parents=True, exist_ok=True)

    sustain = MixedTypeSustain(
        zscore_data=zscore_data,
        z_vals=z_vals,
        z_max=z_max,
        zscore_biomarker_labels=zscore_labels,
        ordinal_prob_nl=ordinal_prob_nl,
        ordinal_prob_score=ordinal_prob_score,
        ordinal_score_vals=ordinal_score_vals,
        ordinal_biomarker_labels=ordinal_labels,
        event_prob_yes=event_prob_yes,
        event_prob_no=event_prob_no,
        event_biomarker_labels=event_labels,
        N_startpoints=N_startpoints,
        N_S_max=N_S_max,
        N_iterations_MCMC=N_iterations_MCMC,
        output_folder=str(output_folder),
        dataset_name=dataset_name,
        use_parallel_startpoints=use_parallel_startpoints,
        seed=seed,
    )

    sustain.run_sustain_algorithm(plot=False)


if __name__ == "__main__":
    main()
