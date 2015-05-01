package dbconn

import (
	"redis/redis"
	"time"
	)

var server = "127.0.0.1:6379" // host:port of server

var Pool = &redis.Pool{
	MaxIdle:     3,
	IdleTimeout: 240 * time.Second,
	Dial: func() (redis.Conn, error) {
		c, err := redis.Dial("tcp", server)

		if err != nil {
			return nil, err
		}
		return c, err
	},

	TestOnBorrow: func(c redis.Conn, t time.Time) error {
		_, err := c.Do("PING")
		return err
	},
}
