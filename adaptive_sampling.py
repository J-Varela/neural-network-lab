
import numpy as np

rng = np.random.default_rng(123)

sigma0 = 0.2

def diffusion_width(alpha, time):
    return np.sqrt(sigma0**2 + 2 * alpha * time)

# Generate candidate parameter combinations
n_candidates = 2000

candidate_alpha = rng.uniform(0.01, 0.2, n_candidates)
candidate_time = rng.uniform(0.1, 5.0, n_candidates)

candidates = np.column_stack((
    candidate_alpha,
    candidate_time,
))

# Generate labels using the physics oracle
candidate_targets = diffusion_width(
    candidate_alpha,
    candidate_time,
)

print("Candidates:", candidates.shape)
print("Targets:", candidate_targets.shape)

# Independent random additions
random_indices = rng.choice(
    n_candidates,
    size=100,
    replace=False,
)

random_X = candidates[random_indices]
random_y = candidate_targets[random_indices]

print("Random additions:", random_X.shape)
print("Random targets:", random_y.shape)



def select_high_error_candidates(
    candidates,
    predictions,
    targets,
    n_select=100,
):
    errors = np.abs(
        predictions.reshape(-1) - targets.reshape(-1)
    )

    # Sort candidates by descending absolute error
    ranked_indices = np.argsort(errors)[::-1]

    selected_indices = ranked_indices[:n_select]

    return (
        candidates[selected_indices],
        targets[selected_indices],
        errors[selected_indices],
    )



# Test the selection algorithm with controlled errors
test_predictions = candidate_targets.copy()

# Introduce artificial prediction errors
test_predictions[:100] += 0.1
test_predictions[100:200] += 0.05

guided_X, guided_y, guided_errors = (
    select_high_error_candidates(
        candidates,
        test_predictions,
        candidate_targets,
        n_select=100,
    )
)

print("\nError-guided selection test:")
print("Selected candidates:", guided_X.shape)
print("Selected targets:", guided_y.shape)
print("Minimum selected error:", guided_errors.min())
print("Maximum selected error:", guided_errors.max())
