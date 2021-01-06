
import pandas as pd
from sklearn import preprocessing
from make_query import make_query_new
from sklearn.preprocessing import StandardScaler

### Load abilities
select_abilities = """
SELECT importance.onetsoc_code,
       importance.title,
       importance.element_id,
       importance.element_name,
       importance.IM,
       level.LV
FROM (SELECT abilities.element_id,
             abilities.onetsoc_code,
             abilities.data_value AS IM,
             content_model_reference.element_name,
             occupation_data.title
       FROM abilities
       JOIN content_model_reference ON abilities.element_id = content_model_reference.element_id
       JOIN occupation_data ON abilities.onetsoc_code = occupation_data.onetsoc_code
       WHERE abilities.scale_id = "IM") AS importance
       JOIN (SELECT abilities.element_id,
                     abilities.onetsoc_code,
                     abilities.data_value AS LV,
                     content_model_reference.element_name,
                     occupation_data.title
             FROM abilities
             JOIN content_model_reference ON abilities.element_id = content_model_reference.element_id
             JOIN occupation_data ON abilities.onetsoc_code = occupation_data.onetsoc_code
             WHERE abilities.scale_id = "LV") AS level
ON importance.element_id = level.element_id AND importance.onetsoc_code = level.onetsoc_code;
"""

### Load skills
select_skills = """
SELECT importance.onetsoc_code,
       importance.title,
       importance.element_id,
       importance.element_name,
       importance.IM,
       level.LV
FROM (SELECT skills.element_id,
             skills.onetsoc_code,
             skills.data_value AS IM,
             content_model_reference.element_name,
             occupation_data.title
      FROM skills
        JOIN content_model_reference ON skills.element_id = content_model_reference.element_id
        JOIN occupation_data ON skills.onetsoc_code = occupation_data.onetsoc_code
        WHERE skills.scale_id = "IM") AS importance
        JOIN (SELECT skills.element_id,
        skills.onetsoc_code,
        skills.data_value AS LV,
        content_model_reference.element_name,
        occupation_data.title
        FROM skills
        JOIN content_model_reference ON skills.element_id = content_model_reference.element_id
        JOIN occupation_data ON skills.onetsoc_code = occupation_data.onetsoc_code
        WHERE skills.scale_id = "LV") AS level
ON importance.element_id = level.element_id AND importance.onetsoc_code = level.onetsoc_code;
"""

### Load knowledge
select_knowledge = """
SELECT importance.onetsoc_code,
       importance.title,
       importance.element_id,
       importance.element_name,
       importance.IM,
       level.LV
FROM (SELECT knowledge.element_id,
             knowledge.onetsoc_code,
             knowledge.data_value AS IM,
             content_model_reference.element_name,
             occupation_data.title
      FROM knowledge
        JOIN content_model_reference ON knowledge.element_id = content_model_reference.element_id
        JOIN occupation_data ON knowledge.onetsoc_code = occupation_data.onetsoc_code
        WHERE knowledge.scale_id = "IM") AS importance
        JOIN (SELECT knowledge.element_id,
        knowledge.onetsoc_code,
        knowledge.data_value AS LV,
        content_model_reference.element_name,
        occupation_data.title
        FROM knowledge
        JOIN content_model_reference ON knowledge.element_id = content_model_reference.element_id
        JOIN occupation_data ON knowledge.onetsoc_code = occupation_data.onetsoc_code
        WHERE knowledge.scale_id = "LV") AS level
        ON importance.element_id = level.element_id AND importance.onetsoc_code = level.onetsoc_code;
"""

### Load work styles
select_workstyle = """

SELECT importance.onetsoc_code,
       importance.title,
       importance.element_id,
       importance.element_name,
       importance.data_value
FROM (SELECT work_styles.element_id,
             work_styles.onetsoc_code,
             work_styles.data_value,
             content_model_reference.element_name,
             occupation_data.title
      FROM work_styles
      JOIN content_model_reference ON work_styles.element_id = content_model_reference.element_id
      JOIN occupation_data ON work_styles.onetsoc_code = occupation_data.onetsoc_code) AS importance
"""

### Load work values
select_workval = """

SELECT importance.onetsoc_code,
       importance.title,
       importance.element_id,
       importance.element_name,
       importance.data_value
FROM (SELECT work_values.element_id,
             work_values.onetsoc_code,
             work_values.data_value,
             content_model_reference.element_name,
             occupation_data.title
      FROM work_values
      JOIN content_model_reference ON work_values.element_id = content_model_reference.element_id
      JOIN occupation_data ON work_values.onetsoc_code = occupation_data.onetsoc_code
      WHERE work_values.scale_id = "EX") AS importance
"""

### load interests
select_interests = """

SELECT importance.onetsoc_code,
       importance.title,
       importance.element_id,
       importance.element_name,
       importance.data_value
FROM (SELECT interests.element_id,
             interests.onetsoc_code,
             interests.data_value,
             content_model_reference.element_name,
             occupation_data.title
      FROM interests
      JOIN content_model_reference ON interests.element_id = content_model_reference.element_id
      JOIN occupation_data ON interests.onetsoc_code = occupation_data.onetsoc_code
      WHERE interests.scale_id = "OI") AS importance

"""

### education
select_ed = """

SELECT level.onetsoc_code,
       level.title,
       level.category,
       level.element_name,
       level.data_value
FROM (SELECT education_training_experience.element_id,
             education_training_experience.onetsoc_code,
             education_training_experience.data_value,
             education_training_experience.category,
             content_model_reference.element_name,
             occupation_data.title
      FROM education_training_experience
      JOIN content_model_reference ON education_training_experience.element_id = content_model_reference.element_id
      JOIN occupation_data ON education_training_experience.onetsoc_code = occupation_data.onetsoc_code
      WHERE education_training_experience.scale_id = "RL") AS level

"""

### Load alternate titles
select_titles = """

SELECT onetsoc_code,
       alternate_title
FROM alternate_titles;

"""

### Load task descriptions
select_tasks = """

SELECT occupation_data.onetsoc_code,
       occupation_data.title,
       task_statements.task
FROM task_statements
JOIN occupation_data ON task_statements.onetsoc_code = occupation_data.onetsoc_code;

"""

### Occupation data
select_occ = """

SELECT onetsoc_code,
       title
FROM occupation_data;

"""

ability = make_query_new(select_abilities, ['SOC', 'Title', 'ElementID', 'ElementName', 'IM', 'LV'])
ability['VALUE'] = ability['IM'].astype('float').pow(2/3) * ability['LV'].astype('float').pow(1/3)
ability_base = ability.pivot(index = 'SOC', columns = 'ElementID', values = 'VALUE')
ability = StandardScaler().fit_transform(ability_base)
ability = pd.DataFrame(ability)
index = pd.DataFrame({'SOC' : ability_base.index})
ability = pd.concat([index, ability], axis = 1)
ability.to_csv('data/ability.csv', index=False)

skills = make_query_new(select_skills, ['SOC', 'Title', 'ElementID', 'ElementName', 'IM', 'LV'])
skills['VALUE'] = skills['IM'].astype('float').pow(2/3) * skills['LV'].astype('float').pow(1/3)
skills_base = skills.pivot(index = 'SOC', columns = 'ElementID', values = 'VALUE')
skills = StandardScaler().fit_transform(skills_base)
skills = pd.DataFrame(skills)
index = pd.DataFrame({'SOC' : skills_base.index})
skills = pd.concat([index, skills], axis = 1)
skills.to_csv('data/skills.csv', index=False)

knowledge = make_query_new(select_knowledge, ['SOC', 'Title', 'ElementID', 'ElementName', 'IM', 'LV'])
knowledge['VALUE'] = knowledge['IM'].astype('float').pow(2/3) * knowledge['LV'].astype('float').pow(1/3)
knowledge_base = knowledge.pivot(index = 'SOC', columns = 'ElementID', values = 'VALUE')
knowledge = StandardScaler().fit_transform(knowledge_base)
knowledge = pd.DataFrame(knowledge)
index = pd.DataFrame({'SOC' : knowledge_base.index})
knowledge = pd.concat([index, knowledge], axis = 1)
knowledge.to_csv('data/knowledge.csv', index=False)

style = make_query_new(select_workstyle, ['SOC', 'Title', 'ElementID', 'ElementName', 'Value'])
style_base = style.pivot(index = 'SOC', columns = 'ElementID', values = 'Value')
style = StandardScaler().fit_transform(style_base)
style = pd.DataFrame(style)
index = pd.DataFrame({'SOC' : style_base.index})
style = pd.concat([index, style], axis = 1)
style.to_csv('data/style.csv', index=False)

values = make_query_new(select_workval, ['SOC', 'Title', 'ElementID', 'ElementName', 'Value'])
values_base = values.pivot(index = 'SOC', columns = 'ElementID', values = 'Value')
values = StandardScaler().fit_transform(values_base)
values = pd.DataFrame(values)
index = pd.DataFrame({'SOC' : values_base.index})
values = pd.concat([index, values], axis = 1)
values.to_csv('data/values.csv', index=False)

interests = make_query_new(select_interests, ['SOC', 'Title', 'ElementID', 'ElementName', 'Value'])
interests_base = interests.pivot(index = 'SOC', columns = 'ElementID', values = 'Value')
interests = StandardScaler().fit_transform(interests_base)
interests = pd.DataFrame(interests)
index = pd.DataFrame({'SOC' : interests_base.index})
interests = pd.concat([index, interests], axis = 1)
interests.to_csv('data/interests.csv', index=False)

ed = make_query_new(select_ed, ['SOC', 'Title', 'ElementID', 'ElementName', 'Value'])
ed_base = ed.pivot(index = 'SOC', columns = 'ElementID', values = 'Value')
ed = StandardScaler().fit_transform(ed_base)
ed = pd.DataFrame(ed)
index = pd.DataFrame({'SOC' : ed_base.index})
ed = pd.concat([index, ed], axis = 1)
ed.to_csv('data/ed.csv', index=False)

titles = make_query_new(select_titles, ['SOC', 'Alt_Title'])
titles.to_csv('data/alt_titles.csv', index = False)

tasks = make_query_new(select_tasks, ['SOC', 'Title', 'Task'])
tasks.to_csv('data/tasks.csv', index = False)

occs = make_query_new(select_occ, ['SOC', 'Title'])
occs.to_csv('data/occs.csv', index = False)
