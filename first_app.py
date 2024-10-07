import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.proportion import proportions_ztest
import streamlit as st


def check_mans(df, days):
    # количество мужчин с пропуском более days дней
    A = len(df[(df['gender'] == 'М') & (df['work_days'] > days)])
    # количество мужчин с пропуском менее days дней
    B = len(df[(df['gender'] == 'М') & (df['work_days'] <= days)])
    # количество женщин с пропуском более days дней
    C = len(df[(df['gender'] == 'Ж') & (df['work_days'] > days)])
    # количество женщин с пропуском менее days дней
    D = len(df[(df['gender'] == 'Ж') & (df['work_days'] <= days)])

    count = np.array([A, C])
    nobs = np.array([A + B, C + D])
    stat, pval = proportions_ztest(count, nobs, alternative='larger')

    fig, axs = plt.subplots(1, 3, figsize=[10, 5], sharex=True, sharey=True)
    axs[0].hist(df.gender[df.work_days <= days], bins=3)
    axs[0].set_title(f"Пропустило {days} и менее дней")
    axs[1].hist(df.gender, bins=3)
    axs[1].set_title(f"Общее соотношение М\Ж")
    axs[2].hist(df.gender[df.work_days > days], bins=3)
    axs[2].set_title(f"Пропустило более{days}")

    if pval > 0.05:
        text = f'Т.к. P-значение равно {pval:.4f}, то мы ***не можем отклонить нулевую гипотезу*** и делаем вывод, что ***нет значимой разницы*** в частоте пропуска более {days} рабочих дней по болезни между мужчинами и женщинами.'
    else:
        text = f'Т.к. P-значение равно {pval:.4f}, то мы ***отклоняем нулевую гипотезу*** и делаем вывод, что действительно Мужчины пропускают более {days}  рабочих дней по болезни в течение года значимо чаще, чем женщины.'
    return text, fig


def check_age(df, days, age):
    # количество работников старше age с пропуском более days дней
    A = len(df[df.age_category == f'>{age}'][df['work_days'] > days])
    # количество работников старше age с пропуском менее days дней
    B = len(df[df.age_category == f'>{age}'][df['work_days'] <= days])
    # количество работников младше age пропуском более days дней
    C = len(df[df.age_category == f'<={age}'][df['work_days'] > days])
    # количество работников младше age с пропуском менее days дней
    D = len(df[df.age_category == f'<={age}'][df['work_days'] <= days])

    fig, axs = plt.subplots(1, 3, figsize=[10, 5], sharex=True, sharey=True)
    axs[0].hist(df.age_category[df.work_days <= days], bins=3)
    axs[0].set_title(f"Пропустило {days} и менее дней")
    axs[1].hist(df.age_category, bins=3)
    axs[1].set_title(f"Общее соотношение категорий")
    axs[2].hist(df.age_category[df.work_days > days], bins=3)
    axs[2].set_title(f"Пропустило более{days}")

    count = np.array([A, C])
    nobs = np.array([A + B, C + D])
    stat, pval = proportions_ztest(count, nobs, alternative='larger')

    if pval > 0.05:
        text = f'Т.к. p-значение равно {pval:.4f}, то не мы можем отклонить нулевую гипотезу и делаем вывод, что нет значимой разницы в частоте пропуска более {days} рабочих дней между заданными категориями возрастов.'
    else:
        text = f'Т.к. p-значение равно {pval:.4f}, то мы отклоняем нулевую гипотезу и делаем вывод, что действительно Работники старше {age} пропускают  в течение года более {days}  рабочих дней  значимо чаще своих более молодых коллег'
    return text, fig


st.title(' Проверка гипотез ')

uploaded_file = st.sidebar.file_uploader("Выберите файл", type=["csv"])
# Проверка, был ли загружен файл
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, encoding="cp1251")
    st.write("Вот данные из CSV файла:")
    st.write(df.head())

    df = df.rename(columns={'Количество больничных дней': 'work_days',
                            'Возраст': 'age',
                            'Пол': 'gender'})
    days = st.sidebar.number_input("Введите количество дней", min_value=min(df.work_days), max_value=max(df.work_days),
                                   value=2)
    age = st.sidebar.number_input("Введите  возраст", min_value=min(df.age), max_value=max(df.age), value=35)
    # Определим категории возрастов
    bins = [0, age, 100]
    labels = [f'<={age}', f'>{age}']
    df['age_category'] = pd.cut(df['age'], bins=bins, labels=labels)

    mans = len(df[df.gender == 'М'])
    women = len(df[df.gender == 'Ж'])
    women_pass_N_days = len(df[(df['work_days'] > days) & (df.gender == 'Ж')])
    mans_pass_N_days = len(df[(df['work_days'] > days) & (df.gender == 'М')])

    less_age = len(df[df.age_category == f'<={age}'])
    more_age = len(df[df.age_category == f'>{age}'])
    lees_pass_N_days = len(df[(df['work_days'] > days) & (df.age_category == f'<={age}')])
    more_pass_N_days = len(df[(df['work_days'] > days) & (df.age_category == f'>{age}')])

    fig, ax = plt.subplots()
    sns.histplot(data=df, x='work_days', hue='gender', bins=max(df.work_days), ax=ax)
    plt.xlabel('Пропущено дней')
    plt.ylabel('Количество людей')
    st.pyplot(fig)
    st.subheader('Первая гипотеза')
    f'Мужчины пропускают в течение года более {days} рабочих дней  по болезни значимо чаще женщин.'
    st.subheader('Вторая гипотеза')
    f'Работники старше {age} лет пропускают в течение года более {days} рабочих дней по болезни значимо чаще своих более молодых коллег.'

    option = st.selectbox("Какую гипотезу хотите проверить?", ('Первая гипотеза', 'Вторая гипотеза'))
    st.write("Вы выбрали:", option)

    if option == 'Первая гипотеза':
        result, fig = check_mans(df, days)
        left_column, right_column = st.columns(2)
        with left_column:
            st.title("Мужчины")
            st.write(f"Всего : {mans} из {mans + women}  ({mans / (mans + women) * 100:.2f}%)")
            st.write(f'{mans_pass_N_days} ({mans_pass_N_days / mans * 100:.2f}%) пропустило более {days} дней')

        with right_column:
            st.title("Женщины")
            st.write(f"Всего : {women} из {mans + women}  ({round(women / (mans + women) * 100, 2)}%)")
            st.write(f'{women_pass_N_days} ({women_pass_N_days / women * 100:.2f}%) пропустило более {days} дней')
        st.write(result)
        st.write(fig)

    if option == 'Вторая гипотеза':
        result, fig = check_age(df, days, age)
        st.title('Возраст')
        left_column, right_column = st.columns(2)
        with left_column:
            st.title(f"{age} и менее")
            st.write(f"Всего : {less_age} из {less_age + more_age}  ({less_age / (less_age + more_age) * 100:.2f}%)")
            st.write(f'{lees_pass_N_days} ({lees_pass_N_days / less_age * 100:.2f}%) пропустило более {days} дней')

        with right_column:
            st.title(f"более {age} ")
            st.write(f"Всего : {more_age} из {less_age + more_age}  ({more_age / (less_age + more_age) * 100:.2f}%)")
            st.write(f'{more_pass_N_days} ({more_pass_N_days / more_age * 100:.2f}%) пропустило более {days} дней')

        st.write(result)
        st.write(fig)


else:
    st.write("Пожалуйста, загрузите файл для отображения данных.")
