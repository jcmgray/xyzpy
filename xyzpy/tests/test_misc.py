from ..misc import (
    sub_split
)


class TestSubSplit:
    def test_2res(self):
        a = [[[('a', 1), ('b', 2)],
              [('c', 3), ('d', 4)]],
             [[('e', 5), ('f', 6)],
              [('g', 7), ('h', 8)]]]
        c, d = sub_split(a)
        assert c.tolist() == [[['a', 'b'],
                               ['c', 'd']],
                              [['e', 'f'],
                               ['g', 'h']]]
        assert d.tolist() == [[[1, 2],
                               [3, 4]],
                              [[5, 6],
                               [7, 8]]]
