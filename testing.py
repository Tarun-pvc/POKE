def calculate_metrics(annotated_keywords, extracted_keywords):
    # Find True Positives, False Positives, and False Negatives
    true_positives = len(set(annotated_keywords) & set(extracted_keywords))
    false_positives = len(set(extracted_keywords) - set(annotated_keywords))
    false_negatives = len(set(annotated_keywords) - set(extracted_keywords))

    # Calculate Precision, Recall, and F1 Score
    precision = true_positives / \
        (true_positives + false_positives) if true_positives + \
        false_positives != 0 else 0
    recall = true_positives / \
        (true_positives + false_negatives) if true_positives + \
        false_negatives != 0 else 0
    f1 = 2 * (precision * recall) / (precision +
                                     recall) if precision + recall != 0 else 0

    return precision, recall, f1


# Example lists (replace these with your actual data)
annotated_keywords = ["keyword1", "keyword2", "keyword3"]
extracted_keywords = ["keyword1", "keyword3",
                      "keyword4", "keyword5", "keyword6"]

# Calculate metrics
precision, recall, f1 = calculate_metrics(
    annotated_keywords, extracted_keywords)

# Print the results
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
