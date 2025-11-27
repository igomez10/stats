package stats

type DecisionTreeNode struct {
	// If the node is a leaf, Value holds the prediction value.
	Feature string
	// If the node is not a leaf, Threshold holds the split threshold.
	Threshold float64
	// Left child node for values <= Threshold.
	Left *DecisionTreeNode
	// Right child node for values > Threshold.
	Right *DecisionTreeNode
	// If the node is a leaf, Value holds the prediction value.
	Value *float64
}

// GetGiniImpurity calculates the Gini impurity for a list of class labels.
// formula Gini = 1 - sum(p_i^2) for all classes i
// where p_i = (number of instances of class i) / (total number of instances)
func GetGiniImpurity(labels []string) float64 {
	if len(labels) == 0 {
		return 0.0
	}

	labelFrequency := make(map[string]int)
	for _, label := range labels {
		labelFrequency[label]++
	}

	summedSquaredProb := 0.0
	for _, labelCount := range labelFrequency {
		prob := float64(labelCount) / float64(len(labels))
		summedSquaredProb += prob * prob
	}

	res := 1.0 - summedSquaredProb
	return res
}
