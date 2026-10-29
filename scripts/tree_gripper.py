"""画像条件付き特徴から、学習済み分類木の指幅を読み出す。"""
import hashlib
import numpy as np
from action_encoding import LOWER_M, UPPER_M


class TreeGripper:
    def __init__(self, document):
        self.document = document
        required = ('children_left', 'children_right', 'feature', 'threshold', 'class_index')
        count = len(document['feature'])
        if document['source_split'] != 'train' or count == 0 or any(len(document[k]) != count for k in required):
            raise ValueError('学習分類木の構造が不正です')
        if document['feature_dim'] <= 0:
            raise ValueError('特徴次元が不正です')
        codebook = np.asarray(document['codebook_m'])
        if codebook.ndim != 1 or not len(codebook) or not np.isfinite(codebook).all() or (codebook < LOWER_M).any() or (codebook > UPPER_M).any():
            raise ValueError('学習指幅辞書が物理範囲外です')
        visited = set()
        def visit(node):
            if not 0 <= node < count or node in visited:
                raise ValueError('分類木が循環または参照範囲外です')
            visited.add(node)
            left = document['children_left'][node]; right = document['children_right'][node]
            if left == right == -1:
                if not 0 <= document['class_index'][node] < len(codebook):
                    raise ValueError('葉の指幅分類が範囲外です')
            else:
                if not 0 <= document['feature'][node] < document['feature_dim'] or not np.isfinite(document['threshold'][node]):
                    raise ValueError('分岐条件が不正です')
                visit(left); visit(right)
        visit(0)
        if len(visited) != count: raise ValueError('未到達の分類木ノードがあります')

    def verify_bindings(self, normalization, weights):
        for field, path in (('normalization_sha256', normalization), ('backbone_weights_sha256', weights)):
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for part in iter(lambda: stream.read(1024*1024), b''): digest.update(part)
            if self.document[field] != digest.hexdigest(): raise ValueError('分類木と動作モデルの識別情報が一致しません')

    def predict(self, features):
        values = np.asarray(features)
        if values.shape != (self.document['feature_dim'],) or not np.isfinite(values).all():
            raise ValueError('分類木の画像・自己状態特徴が不正です')
        node = 0
        while self.document['children_left'][node] != -1:
            axis = self.document['feature'][node]
            branch = 'children_left' if float(values[axis]) <= self.document['threshold'][node] else 'children_right'
            node = self.document[branch][node]
        return self.document['codebook_m'][self.document['class_index'][node]]
