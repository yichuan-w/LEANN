# Unit Test project

This unit test project is based on the [boost unit test framework](https://www.boost.org/doc/libs/1_78_0/libs/test/doc/html/index.html). Below are the simple steps to add new unit test, you could find more usage from the [boost unit test document](https://www.boost.org/doc/libs/1_78_0/libs/test/doc/html/index.html).

## How to add unit test

- Create new [BOOST_AUTO_TEST_SUITE](https://www.boost.org/doc/libs/1_78_0/libs/test/doc/html/boost_test/utf_reference/test_org_reference/test_org_boost_auto_test_suite.html) for each class in an individual cpp file

- Add [BOOST_AUTO_TEST_CASE](https://www.boost.org/doc/libs/1_78_0/libs/test/doc/html/boost_test/utf_reference/test_org_reference/test_org_boost_auto_test_case.html) for each test case in the [BOOST_AUTO_TEST_SUITE](https://www.boost.org/doc/libs/1_78_0/libs/test/doc/html/boost_test/utf_reference/test_org_reference/test_org_boost_auto_test_suite.html)

- Update the [CMakeLists.txt](CMakeLists.txt) file to add the new cpp file to the test project