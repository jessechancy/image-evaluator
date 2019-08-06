def pairwiseloss(outputs, labels):
    if outputs.shape() != labels.shape():
        return
    
