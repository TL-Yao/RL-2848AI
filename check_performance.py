def get_average(logs, episodes_interval):
    grouped_data = {}
    for episode, score, move, loss in logs:
        group_key = (episode - 1) // episodes_interval
        if group_key not in grouped_data:
            grouped_data[group_key] = []
        grouped_data[group_key].append((score, move, loss))

    averages = []
    for group_key in sorted(grouped_data.keys()):
        scores, moves, losses = zip(*grouped_data[group_key])
        avg_score = sum(scores) / len(scores)
        avg_move = sum(moves) / len(moves)
        avg_loss = sum(losses) / len(losses)
        averages.append(
            (group_key * episodes_interval + 1, avg_score, avg_move, avg_loss)
        )

    for average in averages:
        episode, avg_score, avg_move, avg_loss = average
        print(
            f"episode: {episode} - {episode + episodes_interval - 1}, "
            f"Average Score: {avg_score}, Average Move: {avg_move}, Average Loss: {avg_loss}"
        )
