#This query will get 400 StackOverflow posts for the 4 differents groups
#1 - Loop, 2 - Array, 3 - Pointers, 4 - Function


SELECT
  TOP 100
  Posts.Title,
  CASE
    WHEN
      Posts.Title LIKE '%functions%'
      THEN
      'functions'
      END AS Type
FROM
  Posts
WHERE
  Posts.Title LIKE '%functions%'


SELECT
  TOP 100
  Posts.Title,
  CASE
    WHEN
      Posts.Title LIKE '%array%'
      THEN
      'array'
      END AS Type
FROM
  Posts
WHERE
  Posts.Title LIKE '%array%'



SELECT
  TOP 100
  Posts.Title,
  CASE
    WHEN
      Posts.Title LIKE '%pointers%'
      THEN
      'pointers'
      END AS Type
FROM
  Posts
WHERE
  Posts.Title LIKE '%pointers%'



SELECT
  TOP 100
  Posts.Title,
  CASE
    WHEN
      Posts.Title LIKE '%loop%'
      THEN
      'loop'
      END AS Type
FROM
  Posts
WHERE
  Posts.Title LIKE '%loop%'

