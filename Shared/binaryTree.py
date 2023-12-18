class BinaryTreeNode:
    def __init__(self, data):
        self.data = data
        self.leftChild = None
        self.rightChild = None

    def __str__(self, side = "root - ", level=0):
        ret = "\t" * level + side + str(self.data) +"\n"
        if self.leftChild:
            ret += self.leftChild.__str__("left - ", level+1)
        if self.rightChild:
            ret += self.rightChild.__str__("right - ", level+1)
        return ret

def main():    
    bt = BinaryTreeNode({'a': 1, 'b': 2})
    left = bt.leftChild = BinaryTreeNode({'a': 3, 'b': 4})
    right = bt.rightChild = BinaryTreeNode({'a': 5, 'b': 6})
    left.leftChild = BinaryTreeNode({'a': 7, 'b': 8})
    left.rightChild = BinaryTreeNode({'a': 9, 'b': 10})
    rr = right.leftChild = BinaryTreeNode({'a': 11, 'b': 12})
    rr.rightChild = BinaryTreeNode({'a': 13, 'b': 14})
    print(bt)

if __name__ == "__main__":
    main()