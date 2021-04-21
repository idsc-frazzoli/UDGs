from udgs.models.forces_def import IdxParams


def test_indices_struct():
    d1 = dict.fromkeys(IdxParams)
    d2 = dict.fromkeys(IdxParams)

    d1[0]= 10
    d2[5] = 23
    print(d1[0])
    print(d1[1])

    print(d2[5])
    print(d2[6])