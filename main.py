import numpy


def add(x, y):
    """
    Функция вычисляет сумму двух векторов
    """
    return (x + y) % 2


def dot(x, y):
    """
    Функция реализует матричное умножение
    """
    return (x @ y) % 2


def form_verification_matrix_743():
    """
    Функция формирует проверочную матрицу линейного кода (7, 4, 3)
    """
    return [[1, 1, 0],
            [0, 1, 1],
            [1, 0, 1],
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]]


def form_generative_matrix_743():
    """
    Функция формирует порождающую матрицу линейного кода (7, 4, 3)
    """
    return [[1, 0, 0, 0, 1, 1, 0],
            [0, 1, 0, 0, 0, 1, 1],
            [0, 0, 1, 0, 1, 0, 1],
            [0, 0, 0, 1, 0, 0, 0]]


def table_of_syndromes_743(H):
    """
    Функция формирует синдромы линейного кода (7, 4, 3) соответствующие вектору ошибки
    """
    errors = [[1, 0, 0, 0, 0, 0, 0],
              [0, 1, 0, 0, 0, 0, 0],
              [0, 0, 1, 0, 0, 0, 0],
              [0, 0, 0, 1, 0, 0, 0],
              [0, 0, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 0, 1, 0],
              [0, 0, 0, 0, 0, 0, 1]]
    errors = numpy.array(errors)
    syndromes = dot(errors, H)
    table_of_syndromes = {}
    for syndrome, error in zip(syndromes, errors):
        table_of_syndromes[tuple(syndrome)] = error
    return table_of_syndromes


def distance(x, y):
    """
    Функция вычисляет кодовое расстояние между кодовыми словами
    """
    return sum(add(x, y))


def get_vectors_with_min_number_of_ones(length, number_of_ones):
    """
    Функция возвращает генератор строк с заданным минимальным количеством единиц
    """
    if number_of_ones == 0:
        if length == 1:
            yield [0]
            yield [1]
        else:
            for vector in get_vectors_with_min_number_of_ones(length - 1, 0):
                yield vector + [0]
                yield vector + [1]
    elif length == number_of_ones:
        yield [1 for _ in range(length)]
    else:
        for vector in get_vectors_with_min_number_of_ones(length - 1, number_of_ones):
            yield vector + [0]
        for vector in get_vectors_with_min_number_of_ones(length - 1, number_of_ones - 1):
            yield vector + [1]


def form_generative_and_verification_matrix(n, k, d):
    """
    Функция формирует порождающую и проверочнуюю матрицы линейного кода (n, k, d)
    """
    X = [numpy.zeros(n - k, dtype=int)]
    suitable_vectors = [numpy.array(vector, dtype=int) for vector in get_vectors_with_min_number_of_ones(n - k, d - 2)]
    for _ in range(k - 1):
        if len(suitable_vectors) == 0:
            raise Exception('Не получается сформировать линейный код с указанными параметрами')
        vector = suitable_vectors.pop(0)
        X.append(vector)
        new_suitable_vectors = []
        for v in suitable_vectors:
            if distance(v, vector) >= d - 1:
                new_suitable_vectors.append(v)
        suitable_vectors = new_suitable_vectors
    X = numpy.array(X)
    return numpy.concatenate((numpy.eye(k, dtype=int), X), axis=1), \
           numpy.concatenate((X, numpy.eye(n - k, dtype=int)), axis=0)


def generate_errors(n, number_of_errors):
    """
    Функция возвращает генератор ошибок для заданного числа ошибок в кодовом слове
    """
    if n == 1:
        yield [0]
        if number_of_errors == 1:
            yield [1]
    elif number_of_errors == 0:
        yield [0 for _ in range(n)]
    else:
        for err in generate_errors(n - 1, number_of_errors):
            yield err + [0]
        for err in generate_errors(n - 1, number_of_errors - 1):
            yield err + [1]


def get_table_of_syndromes(n, H2, number_of_errors):
    """
    Функция формирует синдромы линейного кода (n, k, d)
    """
    errors = [numpy.array(error) for error in generate_errors(n, number_of_errors)]
    syndromes = dot(numpy.array(errors), H2)
    table_of_syndromes = {}
    for syndrome, error in zip(syndromes, errors):
        table_of_syndromes[tuple(syndrome)] = error
    return table_of_syndromes


def main():
    G = form_generative_matrix_743()
    H = form_verification_matrix_743()
    syndromes = table_of_syndromes_743(H)
    print(syndromes)
    block = [1, 0, 0, 0]
    error = [0, 0, 1, 0, 0, 0, 0]
    word = dot(numpy.array(block), G)
    print('Слово:', word)
    word_with_error = add(word, numpy.array(error))
    print('Слово с ошибкой:', word_with_error)
    syndrome = dot(word_with_error, H)
    print('Синдром: ', syndrome)
    calculated_error = syndromes[tuple(syndrome)]
    print('Найденная ошибка:', calculated_error)
    calculated_word = add(word_with_error, calculated_error)
    print('Исправленное слово:', calculated_word)

    error = [0, 0, 1, 1, 0, 0, 0]
    print('Слово:', word)
    word_with_error = add(word, numpy.array(error))
    print('Слово с ошибкой:', word_with_error)
    syndrome = dot(word_with_error, H)
    print('Синдром: ', syndrome)
    calculated_error = syndromes[tuple(syndrome)]
    print('Найденная ошибка:', calculated_error)
    calculated_word = add(word_with_error, calculated_error)
    print('Исправленное слово:', calculated_word)

    n = 10
    k = 4
    d = 5
    G2, H2 = form_generative_and_verification_matrix(n, k, d)
    block = [0, 1, 1, 1]
    word = dot(numpy.array(block), G2)
    error = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    syndromes = get_table_of_syndromes(n, H2, (d - 1) // 2)
    print(f'Порождающая матрица для кода ({n}, {k}, {d}):')
    print(G2)
    print(f'Проверочная матрица для кода ({n}, {k}, {d}):')
    print(H2)
    print('Слово:', word)
    word_with_error = add(word, numpy.array(error))
    print('Слово с двумя ошибками:', word_with_error)
    syndrome = dot(word_with_error, H2)
    print('Синдром: ', syndrome)
    calculated_error = syndromes[tuple(syndrome)]
    print('Найденные ошибки:', calculated_error)
    calculated_word = add(word_with_error, calculated_error)
    print('Исправленное слово:', calculated_word)

    error = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    print('Слово:', word)
    word_with_error = add(word, numpy.array(error))
    print('Слово с тремя ошибками:', word_with_error)
    syndrome = dot(word_with_error, H2)
    print('Синдром: ', syndrome)
    calculated_error = syndromes[tuple(syndrome)]
    print('Найденные ошибки:', calculated_error)
    calculated_word = add(word_with_error, calculated_error)
    print('Исправленное слово:', calculated_word)


if __name__ == '__main__':
    main()
