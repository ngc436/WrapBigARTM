def test_preprocessing_module():
    from WrapBigARTM.lemmatizaiton_processing import lemmatize_text
    english_data = ['The plane is above the clouds',
                    'Phone 8889980 and order the best борщ']
    result = []
    for test in english_data:
        result.append(lemmatize_text(test))
    assert result == ['plane above cloud',
                      'phone order best']
