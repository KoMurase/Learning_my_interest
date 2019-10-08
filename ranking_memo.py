#Before the normalisation
sns.set_style("white")
sns.set_color_codes(palette='deep')
f, ax = plt.subplots(figsize=(8, 7))
#Check the new distribution
sns.distplot(train['GrLivArea'], color="g");
ax.xaxis.grid(False)
ax.set(ylabel="Frequency")
ax.set(xlabel="GrLivArea")
ax.set(title="GrLivArea distribution") #分布
sns.despine(trim=True, left=True)
plt.show()
#標準化する前の価格の分布の表示

#We use the numpy fuction log1p
train["SalePrice"] = np.log1p(train["SalePrice"])
sns.set_style("white")
sns.set_color_codes(palette='deep')
f, ax = plt.subplots(figsize=(8, 7))
#Check the new distribution
sns.distplot(train['SalePrice'] , fit=norm, color="b");

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(train['SalePrice'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
ax.xaxis.grid(False)
ax.set(ylabel="Frequency")
ax.set(xlabel="SalePrice")
ax.set(title="SalePrice distribution")
sns.despine(trim=True, left=True)

plt.show()


#築年数の処理
train_test["築年数"]=train_test['築年数'].apply(lambda x: x.split('※')[0] if '※____' in x else x)
#~年~か月、~年,~か月があるのでそれを何とかする
#~年ー＞12か月より　*12にしたい
df['価格'] = df['価格'].apply(lambda x: x.strip('億円') + '0000万円' if '億円' in x else x)
df['価格'] = df['価格'].apply(lambda x: str(int(x.split('億')[0])
*10000 + int(x.split('億')[1].split('万円')[0])) + '万円' if '億' in x else x)
#を参考にする
train_test['築年数'] = train_test['築年数'].apply(lambda x: x.split('年')+'*12か月' in x else x)
train_test['築年数'] = trai_test['築年数'].apply(lambda x: str(int(x.split('年')[0])
*12 + int(x.split('年')[1].split('か月')[0])) + 'か月' if '年' in x else x)
#????これでうまくいく中検証要
df['築年数'] = df['築年数'].apply(lambda x: int(float(x.strip('*12か月')) * 12))


#所在階を分数にしたい
