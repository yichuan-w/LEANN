#include <immintrin.h>
#include <math.h>

inline __m256i load_128bit_to_256bit(const __m128i *ptr)
{
	__m128i value128 = _mm_loadu_si128(ptr);
	__m256i value256 = _mm256_castsi128_si256(value128);
	return _mm256_inserti128_si256(value256, _mm_setzero_si128(), 1);
}

float distance_compare_avx512f_f16(const unsigned char *vec1, const unsigned char *vec2, size_t size)
{
	__m512 sum_squared_diff = _mm512_setzero_ps();

	for (int i = 0; i < size / 16; i += 1)
	{
		__m512 v1 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(vec1 + i * 2 * 16)));
		__m512 v2 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(vec2 + i * 2 * 16)));

		__m512 diff = _mm512_sub_ps(v1, v2);
		sum_squared_diff = _mm512_fmadd_ps(diff, diff, sum_squared_diff);
	}

	size_t i = (size / 16) * 16;

	if (i != size)
	{
		__m512 va = _mm512_cvtph_ps(load_128bit_to_256bit((const __m128i *)(vec1 + i * 2)));
		__m512 vb = _mm512_cvtph_ps(load_128bit_to_256bit((const __m128i *)(vec2 + i * 2)));
		__m512 diff512 = _mm512_sub_ps(va, vb);
		sum_squared_diff = _mm512_fmadd_ps(diff512, diff512, sum_squared_diff);
	}

	return _mm512_reduce_add_ps(sum_squared_diff);
}
